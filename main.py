from tkinter.messagebox import NO
from typing import List, Set
from unittest import result
import pandas as pd
import math

DATASET_PATH = "/Users/dmitry/Desktop/agaricus-lepiota.data"

TARGET_COLUMNS = ["cap-shape", "cap-color",
                  "gill-color", "veil-color", "ring-number"]

CLASS_COLUMN = "class"

CLASS_TYPES = ["e", "p"]

# Migrate to main method
data = pd.read_csv(DATASET_PATH, sep=",", header=None)
data.columns = ["class",
                "cap-shape",
                "cap-surface",
                "cap-color",
                "bruises",
                "odor",
                "gill-attachment",
                "gill-spacing",
                "gill-size",
                "gill-color",
                "stalk-shape",
                "stalk-root",
                "stalk-surface-above-ring",
                "stalk-surface-below-ring",
                "stalk-color-above-ring",
                "stalk-color-below-ring",
                "veil-type",
                "veil-color",
                "ring-number",
                "ring-type",
                "spore-print-color",
                "population",
                "habitat"]


class Mushroom:
    def __init__(self, mush_class: str, attributes: dict) -> None:
        self.mush_class = mush_class
        self.attributes = attributes
        pass

    def is_class(self, class_name: str) -> bool:
        return class_name == self.mush_class


class Tree_Leave:
    def __init__(self, parent_branch: str | None, mushrooms: Set[Mushroom], rest_attributes: List[str]) -> None:
        # May be Null if it is root
        self.parent_branch: str | None = parent_branch  # value of parent leave attribute
        self.mushrooms: Set[Mushroom] = mushrooms
        self.rest_attributes: List[str] = rest_attributes
        self.branching_attribute: str | None = None
        self.child_leaves: List[Tree_Leave] = []
        pass

    def is_completed(self) -> bool:
        return self.branching_attribute != None

    # Return None if we don't have rest attributes
    def get_best_branching_attribute(self) -> str | None:
        if self.rest_attributes.count == 0:
            return None

        max_attribute: str | None
        max_gain_ratio: float | None

        for current_attribute in self.rest_attributes:
            current_gain_ratio = get_gain_ration(
                self.mushrooms, current_attribute)

            if (max_attribute == None or max_gain_ratio == None):
                max_attribute = current_attribute
                max_gain_ratio = current_gain_ratio
            else:
                if (math.isclose(max_gain_ratio, current_gain_ratio) or max_gain_ratio < current_gain_ratio):
                    max_attribute = current_attribute
                    max_gain_ratio = current_gain_ratio

        return max_attribute


class Tree:
    def __init__(self, initial_mushrooms: Set[Mushroom], list_of_attributes: List[str]) -> None:
        self.initial_mushrooms: Set[Mushroom] = initial_mushrooms
        self.root_leave: List[Tree_Leave] = Tree_Leave(
            None, initial_mushrooms, list_of_attributes)
        pass

    # Returns false if we cannot split(no rest attributes or empty set)
    def split_leave(self, leave: Tree_Leave) -> bool:
        # Прервать если нарвались на пустое множество грибов
        # Прервать если нарвались на пустое множество оставшихся атрибутов
        
        pass


# Кол-во примеров из мн-ва mushrooms которые принадлежат классу class_type
def get_freq(class_type: str, set_of_mushrooms: Set[Mushroom]) -> int:
    result: int = 0

    for val in set_of_mushrooms:
        if (val.is_class(class_type)):
            result = result + 1


def get_info(set_of_mushrooms: Set[Mushroom]) -> float:
    result: float = 0

    for type in CLASS_TYPES:
        result -= get_freq(type, set_of_mushrooms) / len(set_of_mushrooms) * \
            math.log2(get_freq(type, set_of_mushrooms) / len(set_of_mushrooms))

    return result


def get_splitted_sets_by_attribute(set_of_mushrooms: Set[Mushroom], attribute: str) -> Set[Set[Mushroom]]:
    mushrooms_by_attribute: dict[str, Set[Mushroom]] = {}

    # Take dict<attr_value, set[Mushroom]>
    for mushroom in set_of_mushrooms:
        attribute_value_for_mushroom = mushroom.attributes[attribute]

        # If we not found key(some value of attribute)
        if attribute_value_for_mushroom not in mushrooms_by_attribute:
            mushrooms_by_attribute[attribute_value_for_mushroom] = set()

        mushrooms_by_attribute.get(attribute_value_for_mushroom).add(mushroom)

    result: Set[Set[Mushroom]] = []

    # Put sets of same mushrooms to general set
    for set_of_same_mushrooms in mushrooms_by_attribute.values():
        result.add(set_of_same_mushrooms)

    return result


def get_conditional_info(set_of_mushrooms: Set[Mushroom], splitted_sets_of_mushrooms: Set[Set[Mushroom]]) -> float:
    result: float = 0

    for splitted_set in splitted_sets_of_mushrooms:
        result += len(splitted_set) / len(set_of_mushrooms) * \
            get_info(splitted_set)

    return result


def get_split_estimate(set_of_mushrooms: Set[Mushroom], splitted_sets_of_mushrooms: Set[Set[Mushroom]]) -> float:
    result: float = 0

    for splitted_set in splitted_sets_of_mushrooms:
        result -= len(splitted_set)/len(set_of_mushrooms) * \
            math.log2(len(splitted_set)/len(set_of_mushrooms))

    return result


def get_gain_ration(set_of_mushrooms: Set[Mushroom], atribute: str):
    splitted_sets_of_mushrooms: Set[Set[Mushroom]] = get_splitted_sets_by_attribute(
        set_of_mushrooms, atribute)

    return (get_info(set_of_mushrooms) - get_conditional_info(set_of_mushrooms, splitted_sets_of_mushrooms)) / \
        get_split_estimate(set_of_mushrooms, splitted_sets_of_mushrooms)


def build_tree(tree_node):