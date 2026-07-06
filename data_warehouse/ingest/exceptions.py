"""Exception hierarchy for the Football Data Warehouse."""


class DataWarehouseError(Exception):
    """Base class for all data-warehouse errors."""


class ConfigurationError(DataWarehouseError):
    """Raised when configuration is missing, malformed, or invalid."""


class DownloadError(DataWarehouseError):
    """Raised when a dataset cannot be retrieved from its source."""


class UnknownSourceError(DataWarehouseError):
    """Raised when a requested source name is not registered."""
