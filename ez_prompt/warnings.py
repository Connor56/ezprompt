"""Custom warning types for the ezprompt library."""


class EZPromptWarning(Warning):
    """Base class for all ezprompt warnings."""

    pass


class UnusedInputWarning(EZPromptWarning):
    """Warning raised when template variables are provided but not used in the template."""

    pass
