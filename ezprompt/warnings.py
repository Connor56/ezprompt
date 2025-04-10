"""Custom warning types for the ezprompt library."""


class EzpromptWarning(Warning):
    """Base class for all ezprompt warnings."""

    pass


class UnusedInputWarning(EzpromptWarning):
    """Warning raised when template variables are provided but not used in the template."""

    pass
