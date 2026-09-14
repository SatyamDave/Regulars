"""Data export and deletion (§10 Privacy architecture; GDPR Art. 15/17, CCPA right-to-know/delete).

``regulars.persona.legibility`` already answers "what do you know about me" in plain English.
This package answers the two harder, more formal questions §10 gestures at but never builds:
a portable machine-readable export of everything the system knows about a user
(:mod:`regulars.privacy.export`), and a way to make the system forget them
(:mod:`regulars.privacy.deletion`).
"""

from regulars.privacy.deletion import DeletionReport, delete_user_data
from regulars.privacy.export import DataExport, build_data_export

__all__ = ["DataExport", "DeletionReport", "build_data_export", "delete_user_data"]
