from uer.targets.mlm_target import MlmTarget
from uer.targets.sp_target import SpTarget
from uer.targets.lm_target import LmTarget
from uer.targets.cls_target import ClsTarget
from uer.targets.bilm_target import BilmTarget
from uer.targets.target import Target
from uer.targets.mlm_edit_target import MlmEditTarget

str2target = {"sp": SpTarget, "mlm": MlmTarget, "lm": LmTarget,
              "bilm": BilmTarget, "cls": ClsTarget, "edit":MlmEditTarget}

__all__ = ["Target", "SpTarget", "MlmTarget", "LmTarget", "BilmTarget", "ClsTarget", "MlmEditTarget", "str2target"]
