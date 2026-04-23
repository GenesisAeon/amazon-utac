"""Template registry — add a new module here to register a new template."""

from diamond_setup._types import TemplateDict

from .genesis import TEMPLATE as GENESIS_TEMPLATE
from .minimal import TEMPLATE as MINIMAL_TEMPLATE
from .amazon_utac import TEMPLATE as AMAZON_UTAC_TEMPLATE

REGISTRY: dict[str, TemplateDict] = {
    "minimal": MINIMAL_TEMPLATE,
    "genesis": GENESIS_TEMPLATE,
    "amazon-utac": AMAZON_UTAC_TEMPLATE,
}

__all__ = ["REGISTRY"]
