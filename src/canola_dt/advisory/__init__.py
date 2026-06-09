"""Advisory (decision-support) layer of the canola digital twin.

Couples Canola Council agronomic alert logic with the calibrated biophysical yield
from the process model. See ``scripts/run_advisory.py`` for an end-to-end demo.
"""

from canola_dt.advisory.agronomy import (
    AgronomyParameters,
    AlertSeverity,
    CultivarType,
    GrowthStage,
    PrecedingCrop,
    Species,
)
from canola_dt.advisory.engine import (
    CanolaAdvisoryEngine,
    calculate_seeding_rate,
    estimate_n_requirement,
    get_harvest_strategy,
)
from canola_dt.advisory.state import Alert, CanolaFieldState
from canola_dt.advisory.wheat_agronomy import (
    WheatAgronomyParameters,
    WheatClass,
    WheatGrowthStage,
    WheatPrecedingCrop,
)
from canola_dt.advisory.wheat_engine import (
    WheatAdvisoryEngine,
    wheat_n_requirement,
    wheat_seeding_rate,
)
from canola_dt.advisory.wheat_state import WheatFieldState
from canola_dt.advisory.barley_agronomy import (
    BarleyAgronomyParameters,
    BarleyGrowthStage,
    BarleyPrecedingCrop,
    BarleyType,
)
from canola_dt.advisory.barley_engine import BarleyAdvisoryEngine, barley_seeding_rate
from canola_dt.advisory.barley_state import BarleyFieldState
from canola_dt.advisory.pea_agronomy import (
    PeaAgronomyParameters,
    PeaGrowthStage,
    PeaPrecedingCrop,
    PeaType,
)
from canola_dt.advisory.pea_engine import PeaAdvisoryEngine, pea_seeding_rate
from canola_dt.advisory.pea_state import PeaFieldState

__all__ = [
    # canola
    "AgronomyParameters", "AlertSeverity", "CultivarType", "GrowthStage",
    "PrecedingCrop", "Species", "Alert", "CanolaFieldState",
    "CanolaAdvisoryEngine", "calculate_seeding_rate", "estimate_n_requirement",
    "get_harvest_strategy",
    # wheat
    "WheatAgronomyParameters", "WheatClass", "WheatGrowthStage", "WheatPrecedingCrop",
    "WheatFieldState", "WheatAdvisoryEngine", "wheat_seeding_rate", "wheat_n_requirement",
    # barley
    "BarleyAgronomyParameters", "BarleyType", "BarleyGrowthStage", "BarleyPrecedingCrop",
    "BarleyFieldState", "BarleyAdvisoryEngine", "barley_seeding_rate",
    # pea
    "PeaAgronomyParameters", "PeaType", "PeaGrowthStage", "PeaPrecedingCrop",
    "PeaFieldState", "PeaAdvisoryEngine", "pea_seeding_rate",
]
