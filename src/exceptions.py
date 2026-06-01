"""Custom exceptions for the face recognition system."""


class FaceRecognitionError(Exception):
    """Base exception for the application."""


class DatabaseError(FaceRecognitionError):
    """Database operation failed."""


class CameraError(FaceRecognitionError):
    """Camera operation failed."""


class RegistrationError(FaceRecognitionError):
    """Face registration failed."""


class RecognitionError(FaceRecognitionError):
    """Face recognition failed."""


class ModelNotFoundError(FaceRecognitionError):
    """Trained model file not found."""


class TrainingError(FaceRecognitionError):
    """Model training failed."""


class ConfigurationError(FaceRecognitionError):
    """Configuration error."""
