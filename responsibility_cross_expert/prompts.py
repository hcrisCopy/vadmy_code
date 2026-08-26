"""Published event-prompt dictionaries for UCF-Crime and XD-Violence.

Abnormal prompts are copied from Tables A1/A2 of LAP (TOMM 2026):
https://doi.org/10.1145/3801554

Normal prompts are copied from LaGoVAD's ``DatasetSpecVerbalizer`` because LAP
only publishes abnormal-event prompts. No prompt here was generated for this
project.
"""

from __future__ import annotations


LAP_REFERENCE = "Tao et al., LAP, TOMM 2026, Tables A1/A2"
LAGOVAD_REFERENCE = "Liu et al., LaGoVAD, ICLR 2026, DatasetSpecVerbalizer"

NORMAL = [
    "Normal",
    "Normal user uploaded video",
    "Normal surveillance footage.",
    "Normal behavior, views or surveillance content captured by camera that not involve any unexpected or unusual events.",
]

UCF_FIGHTING = [
    "A group of people are fighting", "Two people are fighting fiercely",
    "A person is kicking another person", "A person is punching someone",
    "Someone is beating a person on the ground", "People are chasing and hitting someone",
    "A person is pushing another person violently",
]
UCF_SHOOTING = [
    "A man is shooting a gun", "A person is firing a weapon",
    "A man is holding a gun to rob someone", "A person is aiming a pistol",
    "Someone is throwing a knife",
]
UCF_THEFT = [
    "A person is breaking into a house", "A thief is stealing items from a shelf",
    "Someone is taking money from the cashier", "A person is snatching a bag",
    "People are breaking a window to enter", "A person is putting stolen goods in a pocket",
    "Masked men are robbing a store",
]
UCF_ACCIDENT = [
    "A car crashes into another car", "A car hits a pedestrian",
    "A vehicle overturns on the road", "Two vehicles collide at an intersection",
    "A truck crashes into a building",
]
UCF_FIRE = [
    "A fire is burning", "Smoke is rising from a fire",
    "A large explosion occurs", "Something explodes with fire",
]
UCF_VANDALISM = [
    "A person is destroying public property", "Someone is smashing a car window",
    "A group is throwing stones",
]
UCF_ARREST = [
    "Police are arresting a suspect", "A person is being handcuffed by police",
]

XD_FIGHTING_ABUSE = [
    "Two people are fighting", "A person is kicking another person",
    "A group of people are arguing", "A man is throwing something to a man",
    "A group of people are fighting", "A person is punching someone",
    "Someone is beating a person on the ground", "People are chasing and hitting someone",
    "A man is slapping a woman",
]
XD_SHOOTING = [
    "A man is shooting a gun", "A man is throwing a knife",
    "A person is firing a rifle", "A man is holding a pistol",
    "Gunmen are shooting in the street", "A person is aiming a weapon",
    "Someone is wielding a machete",
]
XD_ACCIDENT = [
    "A person is getting hit by a car", "A car crashes into a vehicle",
    "A vehicle overturns on the road", "A truck hits a pedestrian",
    "Two vehicles collide at high speed", "A car crashes into a guardrail",
]
XD_EXPLOSION = [
    "A fire is burning", "A large explosion occurs",
    "A bomb explodes with smoke", "A building is blown up",
]
XD_RIOT = [
    "A crowd is throwing stones", "People are smashing car windows",
    "A mob is destroying property", "Protesters are clashing with police",
]

PROMPT_BANKS: dict[str, dict[str, list[str]]] = {
    "ucf": {
        "Normal": NORMAL,
        "Abuse": UCF_FIGHTING, "Arrest": UCF_ARREST, "Arson": UCF_FIRE,
        "Assault": UCF_FIGHTING, "Burglary": UCF_THEFT, "Explosion": UCF_FIRE,
        "Fighting": UCF_FIGHTING, "RoadAccidents": UCF_ACCIDENT,
        "Robbery": UCF_THEFT, "Shooting": UCF_SHOOTING,
        "Shoplifting": UCF_THEFT, "Stealing": UCF_THEFT,
        "Vandalism": UCF_VANDALISM,
    },
    "xd": {
        "normal": NORMAL, "fighting": XD_FIGHTING_ABUSE,
        "shooting": XD_SHOOTING, "riot": XD_RIOT,
        "abuse": XD_FIGHTING_ABUSE, "car accident": XD_ACCIDENT,
        "explosion": XD_EXPLOSION,
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
    normal = normal_class_name(dataset)
    return [name for name in class_names(dataset) if name != normal]


def normal_class_name(dataset: str) -> str:
    return "Normal" if dataset == "ucf" else "normal"


def label_targets(dataset: str, label: str) -> list[int]:
    names = abnormal_class_names(dataset)
    raw = [label] if dataset == "ucf" else str(label).split("-")
    mapped = [LABEL_MAPS[dataset][value] for value in raw if value in LABEL_MAPS[dataset]]
    return [names.index(value) for value in mapped if value in names]


def prompt_provenance() -> dict[str, str]:
    return {"abnormal": LAP_REFERENCE, "normal": LAGOVAD_REFERENCE}
