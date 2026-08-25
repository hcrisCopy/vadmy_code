"""Dataset prompt banks copied from LaGoVAD's DatasetSpecVerbalizer.

Only visually observable descriptions for UCF-Crime and XD-Violence are
retained.  No new LLM-generated description is introduced here.
"""

from __future__ import annotations


NORMAL = [
    "Normal",
    "Normal user uploaded video",
    "Normal surveillance footage.",
    "Normal behavior, views or surveillance content captured by camera that not involve any unexpected or unusual events.",
]

FIGHTING = [
    "Fighting",
    "Violence",
    "Using violence to injure or kill someone, usually involving group fights.",
    "A group of people fighting and brawling, which can be seen in punches, kicks.",
    "In sports, players have conflicts and start fighting each other.",
    "A man knocked down another person to the ground.",
]

SHOOTING = [
    "Shooting",
    "Firing a weapon, usually involving muzzle flash and the trajectory of the bullets",
    "The act of firing a firearm, often with gun flame and people lying down.",
    "A person points a gun at another person and shoots, and the muzzle emits flames and smoke.",
]

RIOT = [
    "Riot scene",
    "The chaotic riot scene. There are many people and special police officers who are suppressing it.",
    "large-scale, public riot, often involving breaking windows, setting fires, and clashing with law enforcement.",
    "Riot scene, armed police wearing helmets and holding shields forming a human wall, with smoke and flames in the background.",
]

CAR_ACCIDENT = [
    "A collision between two or more vehicles, often resulting in injury or damage.",
    "Traffic accident scene, vehicle colliding with pedestrian",
    "The driving recorder recorded that two cars collided with each other",
    "Road accident scene",
]

EXPLOSION = [
    "Explosion",
    "Explosion, often resulting in fire, smoke, and scattered debris.",
    "The scene of the explosion, with mushroom clouds and smoke after the explosion.",
]


PROMPT_BANKS: dict[str, dict[str, list[str]]] = {
    "ucf": {
        "Normal": NORMAL,
        "Abuse": [
            "Intentional beating or abuse of animals like dogs.",
            "Abuse, ill-treatment of pets like dogs or cats.",
            "Beating or kicking pets, torture of animals",
        ],
        "Arrest": ["Police arresting suspects, which may involve pressing them to the ground, controlling hands or aiming with guns."],
        "Arson": ["The deliberate setting of a fire by someone, usually characterized by flames, smoke, puring gasoline."],
        "Assault": [
            "Assault",
            "The man unilaterally attacked the victim, using fists or sticks",
            "Multiple people surround and assault one person with fists and cudgels.",
            "Continuous assault by one person on another",
        ],
        "Burglary": ["Burglary, usually characterized by crossing the cashier, breaking doors and windows, and carry things."],
        "Explosion": EXPLOSION,
        "Fighting": FIGHTING,
        "RoadAccidents": CAR_ACCIDENT,
        "Robbery": ["Robbing others property through violent means such as beating or holding a gun"],
        "Shooting": SHOOTING,
        "Shoplifting": ["Shoplifting, sneak things into bags, clothes or under skirts in stores."],
        "Stealing": ["Stealing property from cars or stealing motorcycles and batteries."],
        "Vandalism": ["Damaging vehicles, overturning shelves, or smashing store door."],
    },
    "xd": {
        "normal": NORMAL,
        "fighting": FIGHTING,
        "shooting": SHOOTING,
        "riot": RIOT,
        "abuse": [
            "Abuse, ill-treatment of people.",
            "Someone suffering from physical abuse",
            "A person is being abused by others.",
        ],
        "car accident": CAR_ACCIDENT,
        "explosion": EXPLOSION,
    },
}

LABEL_MAPS = {
    "ucf": {name: name for name in PROMPT_BANKS["ucf"]},
    "xd": {
        "A": "normal", "B1": "fighting", "B2": "shooting", "B4": "riot",
        "B5": "abuse", "B6": "car accident", "G": "explosion",
    },
}


def class_names(dataset: str) -> list[str]:
    return list(PROMPT_BANKS[dataset])


def abnormal_class_names(dataset: str) -> list[str]:
    normal = "Normal" if dataset == "ucf" else "normal"
    return [name for name in class_names(dataset) if name != normal]


def label_targets(dataset: str, label: str) -> list[int]:
    names = abnormal_class_names(dataset)
    raw = [label] if dataset == "ucf" else str(label).split("-")
    mapped = [LABEL_MAPS[dataset][value] for value in raw if value in LABEL_MAPS[dataset]]
    return [names.index(value) for value in mapped if value in names]
