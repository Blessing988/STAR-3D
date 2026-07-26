"""Shared constants for the PhysicalAI Track 1 task."""

CLASS_TO_ID = {
    "Person": 0,
    "Forklift": 1,
    "NovaCarter": 2,
    "Transporter": 3,
    "FourierGR1T2": 4,
    "AgilityDigit": 5,
    "PalletTruck": 6,
}

ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}

GT_KEY_ALIASES = {
    "object_type": ("object_type", "object type"),
    "object_id": ("object_id", "object id"),
    "location": ("3d_location", "3d location"),
    "scale": ("3d_bounding_box_scale", "3d bounding box scale"),
    "rotation": ("3d_bounding_box_rotation", "3d bounding box rotation"),
    "bbox2d": ("2d_bounding_box_visible", "2d bounding box visible"),
}

FPS = 30

