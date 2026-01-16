from enum import Enum


class ActionSwapType(Enum):
    Original = "original_video"
    ActionSwap = "actionswap_video"
    PersonInpainting = "person_inpainting_video"
    PersonOnly = "person_only_video"
