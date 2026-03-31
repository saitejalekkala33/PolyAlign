from __future__ import annotations

from polyalign_data.datasets.cmrc2018 import CMRC2018Formatter
from polyalign_data.datasets.coig_cqia import COIGCQIAFormatter
from polyalign_data.datasets.coqa import CoQAFormatter
from polyalign_data.datasets.dailydialog import DailyDialogFormatter
from polyalign_data.datasets.dolly import DollyFormatter
from polyalign_data.datasets.drcd import DRCDFormatter
from polyalign_data.datasets.dureader import DuReaderFormatter
from polyalign_data.datasets.eli5_category import ELI5CategoryFormatter
from polyalign_data.datasets.hc3_chinese import HC3ChineseFormatter
from polyalign_data.datasets.ms_marco import MSMARCOFormatter
from polyalign_data.datasets.multiwoz import MultiWOZFormatter
from polyalign_data.datasets.natural_questions import NaturalQuestionsFormatter
from polyalign_data.datasets.oasst2_zh import OASST2ChineseFormatter
from polyalign_data.datasets.squad_v2 import SQuADV2Formatter


FORMATTERS = {
    "coqa": CoQAFormatter,
    "cmrc2018": CMRC2018Formatter,
    "coig_cqia": COIGCQIAFormatter,
    "dailydialog": DailyDialogFormatter,
    "dolly": DollyFormatter,
    "drcd": DRCDFormatter,
    "dureader": DuReaderFormatter,
    "eli5": ELI5CategoryFormatter,
    "eli5_category": ELI5CategoryFormatter,
    "hc3_chinese": HC3ChineseFormatter,
    "ms_marco": MSMARCOFormatter,
    "multiwoz": MultiWOZFormatter,
    "natural_questions": NaturalQuestionsFormatter,
    "nq": NaturalQuestionsFormatter,
    "oasst2_zh": OASST2ChineseFormatter,
    "squad_v2": SQuADV2Formatter,
}


def canonical_dataset_names() -> list[str]:
    return [
        "dolly",
        "ms_marco",
        "coqa",
        "eli5_category",
        "squad_v2",
        "natural_questions",
        "dailydialog",
        "multiwoz",
    ]


def canonical_chinese_dataset_names() -> list[str]:
    return [
        "oasst2_zh",
        "dureader",
        "cmrc2018",
        "drcd",
        "hc3_chinese",
        "coig_cqia",
    ]


def create_formatter(name: str, *, seed: int, cache_dir: str):
    if name not in FORMATTERS:
        available = ", ".join(sorted(FORMATTERS))
        raise KeyError(f"Unknown dataset '{name}'. Available datasets: {available}")
    formatter_cls = FORMATTERS[name]
    return formatter_cls(seed=seed, cache_dir=cache_dir)
