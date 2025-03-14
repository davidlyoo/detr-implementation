import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """
    모델 예측 Query와 Ground Truth 간의 1:1 매칭을 수행
    Hungarian Algorithm을 사용하여 
    - 각 cost 가중치 설정
    """
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """
        Args:
            cost_class: 클래스 분류 비용 가중치
            cost_bbox: 바운딩 박스 L1 비용 가중치
            cost_giou: GIoU 비용 가중치
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all cost can't be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """_summary_

        Args:
            outputs(dict): DETR 예측값
                "pred_logits": [batch_size, num_queries, num_classes] -> Query의 클래스 확률
                "pred_boxes": [batch_size, num_queries, 4] -> Query의 바운딩 박스 좌표 (cx, cy, w, h)

            targets(list): GT 정보보
                "labels": [num_target_boxes] -> GT 객체 클래스 라벨벨
                "boxes": [num_target_boxes, 4] -> GT 객체 바운딩 박스 좌표

        Returns:
            list
                - `index_i`: 선택된 Query의 인덱스
                - `index_j`: 매칭된 GT의 인덱스
                - `len(index_i) == len(index_j) == min(num_queries, num_gt_boxes)`
        """
        # 배치 크기, Query 개수 가져오기
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # Query 예측값 변환
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1) # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1) # [batch_size * num_queries, 4]
        
        # GT 데이터 변환 - 배치 내 모든 GT의 클래스 라벨, 바운딩 박스를 하나의 텐서로 결합합
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        
        # Query-GT 간 비용 계산
        # 클래스 비용 계산산
        cost_class = -out_prob[:, tgt_ids]
        
        # 바운딩 박스 L1 거리 비용 계산산
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        
        # GIoU 비용 계산6
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        
        # 최종 비용 행렬 생성
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        
        # Hungarian Algorithm 적용 - 최적 매칭 수행행
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in  enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]



def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)