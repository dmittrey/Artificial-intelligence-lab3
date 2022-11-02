from typing import Dict, List
import pandas as pd
import math
from queue import Queue
import seaborn as sns
import matplotlib.pyplot as plt

DATASET_PATH = "/Users/dmitry/Desktop/agaricus-lepiota.data"

TARGET_COLUMNS = ["cap-shape", "cap-color",
                  "gill-color", "veil-color", "ring-number"]

CLASS_COLUMN = "class"

CLASS_TYPES = ["e", "p"]

# Migrate to main method
data = pd.read_csv(DATASET_PATH, sep=",", header=0)
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


class Tree_Leave:
    def __init__(self, parent_branch: str | None, mushrooms: List[Mushroom], rest_attributes: List[str]) -> None:
        # May be Null if it is root
        self.parent_branch: str | None = parent_branch  # value of parent leave attribute
        self.mushrooms: List[Mushroom] = mushrooms
        self.rest_attributes: List[str] = rest_attributes
        self.branching_attribute: str | None = None
        self.child_leaves: List[Tree_Leave] = list()
        pass

    # Return None if we don't have rest attributes
    def get_best_branching_attribute(self) -> str | None:
        if self.rest_attributes.count == 0:
            return None

        max_attribute: str | None = None
        max_gain_ratio: float | None = None

        for current_attribute in self.rest_attributes:
            splitted_sets_of_mushrooms: Dict[str, List[Mushroom]] = get_splitted_sets_by_attribute(
                self.mushrooms, current_attribute)

            current_gain_ratio = get_gain_ration(
                self.mushrooms, splitted_sets_of_mushrooms)

            if (max_attribute == None or max_gain_ratio == None):
                max_attribute = current_attribute
                max_gain_ratio = current_gain_ratio
            else:
                if (math.isclose(max_gain_ratio, current_gain_ratio) or max_gain_ratio < current_gain_ratio):
                    max_attribute = current_attribute
                    max_gain_ratio = current_gain_ratio

        return max_attribute

    # 1) Find best attribute
    # 2) Create n leaves
    # 3) Fill splitted sets in leaves
    # 4) Return list of created leaves or None(no rest attributes or empty set)
    def split_leave(self) -> List | None:
        # Чтобы найти лучший атрибут
        best_attribute: str | None = self.get_best_branching_attribute()

        if (best_attribute != None):
            if (len(self.mushrooms) != 0):
                # Set branching attribute
                self.branching_attribute = best_attribute

                # Alloc list for childs
                self.child_leaves: List[Tree_Leave] = []

                # Found splitted sets of mushrooms by attribute
                splitted_sets_of_mushrooms: Dict[str, List[Mushroom]] = get_splitted_sets_by_attribute(
                    self.mushrooms, best_attribute)

                # Append child leaves
                self.rest_attributes.remove(best_attribute)
                for splitted_set_attr, splitted_set_val in splitted_sets_of_mushrooms.items():
                    self.child_leaves.append(Tree_Leave(
                        splitted_set_attr, splitted_set_val, self.rest_attributes))

                return self.child_leaves

            else:
                # Cannot split because set of mushrooms is empty
                return None
        else:
            # Cannot split because no rest attributes
            return None

    # Выводит проценты для каждого класса
    def predict_proba(self, classes: List[str]) -> Dict[str, float]:
        class_predictions: Dict[str, float] = dict()

        # Start of counter
        for mushroom_class in classes:
            class_predictions[mushroom_class] = 0

        for mushroom in self.mushrooms:
            class_predictions[mushroom.mush_class] += 1

        # Нормируем
        for predict in class_predictions.values():
            predict = predict / len(self.mushrooms)

        return class_predictions

    # would output class 0 since the class probability for 0 is 0.6
    # Выводит наиболее вероятный класс
    def predict(self, classes: List[str]) -> str | None:
        class_predictions: Dict[str, float] = self.predict_proba(classes)

        max_populate_class: str | None = None
        max_populate_probe: float = 0

        for predict_key, predict_value in class_predictions.items():
            if (predict_value > max_populate_probe):
                max_populate_probe = predict_value
                max_populate_class = predict_key

        return max_populate_class

    # Returns target tree leave
    def decide(self, mushroom: Mushroom):
        mushroom_attr_val = mushroom.attributes[self.branching_attribute]

        for child in self.child_leaves:
            if child.parent_branch == mushroom_attr_val:
                return child

        # TODO А что делать если не нашли?(Может самое близкое значение?)
        return None


class Tree:
    def __init__(self, initial_mushrooms: List[Mushroom], list_of_attributes: List[str]) -> None:
        self.initial_mushrooms: List[Mushroom] = initial_mushrooms
        self.root_leave: List[Tree_Leave] = Tree_Leave(
            None, initial_mushrooms, list_of_attributes)
        self.terminate_leaves: List[Tree_Leave] = list()
        pass

    # Return status of building
    def build_tree(self) -> List[Tree_Leave]:
        q: Queue[Tree_Leave] = Queue()
        terminate_leaves = []  # Leaves that we cannot split

        # Put root in queue
        q.put(self.root_leave)

        while (not q.empty()):
            current_leave = q.get()

            new_leaves = current_leave.split_leave()

            if new_leaves == None:
                terminate_leaves.append(current_leave)
            else:
                for leave in new_leaves:
                    q.put(leave)

        return terminate_leaves

    # Returns mushroom class or None if we don't create decision tree before
    def decide(self, mushroom: Mushroom, classes: List[str]) -> Tree_Leave | None:
        if (self.terminate_leaves.count == 0):
            return None

        current_leave: Tree_Leave = self.root_leave

        while (current_leave.branching_attribute != None):
            current_leave = current_leave.decide(mushroom)

        return current_leave


# Amount of examples from mushrooms set that belongs to class_type
def get_freq(class_type: str, set_of_mushrooms: List[Mushroom]) -> int:
    result: int = 0

    for val in set_of_mushrooms:
        if (val.mush_class == class_type):
            result = result + 1

    return result

# Если мы нашли такой атрибут, чтобы не разбивать выборку, то кол-во информации минимально


def get_info(set_of_mushrooms: List[Mushroom]) -> float:
    result: float = 0

    for type in CLASS_TYPES:
        freq_for_class = get_freq(type, set_of_mushrooms)
        if (freq_for_class != 0):
            result -= freq_for_class / len(set_of_mushrooms) * \
                math.log2(freq_for_class / len(set_of_mushrooms))

    return result


def get_splitted_sets_by_attribute(set_of_mushrooms: List[Mushroom], attribute: str) -> Dict[str, List[Mushroom]]:
    mushrooms_by_attribute: Dict[str, List[Mushroom]] = dict()

    # Take dict<attr_value, set[Mushroom]>
    for mushroom in set_of_mushrooms:
        attribute_value_for_mushroom = mushroom.attributes[attribute]

        # If we not found key(some value of attribute)
        if attribute_value_for_mushroom not in mushrooms_by_attribute:
            mushrooms_by_attribute[attribute_value_for_mushroom] = list()

        mushrooms_by_attribute.get(
            attribute_value_for_mushroom).append(mushroom)

    return mushrooms_by_attribute

    # result: List[Set[Mushroom]] = list()

    # # Put sets of same mushrooms to general set
    # for set_of_same_mushrooms in mushrooms_by_attribute.values():
    #     result.append(set_of_same_mushrooms)

    # return result


def get_conditional_info(set_of_mushrooms: List[Mushroom], splitted_sets_of_mushrooms: Dict[str, List[Mushroom]]) -> float:
    result: float = 0

    for splitted_set in splitted_sets_of_mushrooms.values():
        result += len(splitted_set) / len(set_of_mushrooms) * \
            get_info(splitted_set)

    return result


def get_split_estimate(set_of_mushrooms: List[Mushroom], splitted_sets_of_mushrooms: Dict[str, List[Mushroom]]) -> float:
    result: float = 0

    # print("\n\n")

    for splitted_set in splitted_sets_of_mushrooms.values():
        result -= len(splitted_set)/len(set_of_mushrooms) * \
            math.log2(len(splitted_set)/len(set_of_mushrooms))

        # print(len(splitted_set))
        # print(len(set_of_mushrooms))
        # print(result)

    return result


def get_gain_ration(set_of_mushrooms: List[Mushroom], splitted_sets_of_mushrooms: Dict[str, List[Mushroom]]) -> float:
    info = get_info(set_of_mushrooms)
    conditional_info = get_conditional_info(
        set_of_mushrooms, splitted_sets_of_mushrooms)
    split_estimate = get_split_estimate(
        set_of_mushrooms, splitted_sets_of_mushrooms)  # TODO  Что делать когда тут 0?

    if (split_estimate == 0):
        return 0
    else:
        return (info - conditional_info) / split_estimate


class Metrics:
    def __init__(self, true_positive_counts: int, false_positive_counts: int, false_negative_counts: int, true_negative_counts: int) -> None:
        self.true_positive_counts = true_positive_counts
        self.false_positive_counts = false_positive_counts
        self.false_negative_counts = false_negative_counts
        self.true_negative_counts = true_negative_counts
        pass

    # Просто доля верно предсказанных попаданий
    def get_accuracy(self) -> float:
        return (self.true_positive_counts + self.true_negative_counts) / \
            (self.true_positive_counts + self.true_negative_counts +
             self.false_positive_counts + self.false_negative_counts)

    # Метрика точности
    # Доля объектов, названных и являющихся положительными
    def get_precision(self) -> float:
        return (self.true_positive_counts) / (self.true_positive_counts + self.false_positive_counts)

    # Метрика полноты
    # Доля обхектов положительного класса, которую удалось найти
    def get_recall(self) -> float:
        return (self.true_positive_counts) / (self.true_positive_counts + self.false_negative_counts)

    def get_tpr(self) -> float:
        return (self.true_positive_counts / (self.true_positive_counts + self.false_negative_counts))

    def get_fpr(self) -> float:
        return (self.false_positive_counts / (self.false_positive_counts + self.true_negative_counts))


def get_metrics(decision_tree: Tree, samples: List[Mushroom], classes: List[str]) -> Metrics:
    true_positive_count: int = 0
    false_negative_count: int = 0
    false_positive_count: int = 0
    true_negative_count: int = 0

    for sample in samples:
        decided_class: Tree_Leave = decision_tree.decide(
            sample, classes).predict(classes)

        if (decided_class == "e" and sample.mush_class == "e"):
            true_positive_count += 1
        if (decided_class == "e" and sample.mush_class == "p"):
            false_positive_count += 1
        if (decided_class == "p" and sample.mush_class == "p"):
            true_negative_count += 1
        if (decided_class == "p" and sample.mush_class == "e"):
            false_negative_count += 1

    return Metrics(true_positive_count, false_positive_count, false_negative_count, true_negative_count)

# 1) Сортировка вероятностей для положительного класса в порядке убывания
# 2) Обрабатываем по одному экземпляру за раз
# 3) Считаем TPR и FPR

# TPR = True Positives / All Positives
# FPR = False Positives / All negatives


def paint_AUC_ROC(decision_tree: Tree, samples: List[Mushroom], classes: List[str]) -> None:
    coords = [(0, 0)]
    lower_index: float = 1

    true_positive_count: int = 0
    false_positive_count: int = 0
    false_negative_count: int = 0

    positive_count: int = 0
    negative_count: int = 0

    while lower_index > 0:
        for sample in samples:
            decided_probes = decision_tree.decide(
                sample, classes).predict_proba(classes)

            if (decided_probes["e"] > lower_index):
                decided_class = "e"
            else:
                decided_class = "p"

            if (decided_class == "e" and sample.mush_class == "e"):
                true_positive_count += 1
            if (decided_class == "e" and sample.mush_class == "p"):
                false_positive_count += 1
            
            if (sample.mush_class == "e"):
                positive_count += 1
            else:
                negative_count += 1

            coords.append((true_positive_count, false_positive_count))
        
            lower_index -= 0.1

    # Запаковываем по парам значения положительно и отрицательно найденных
    fp, tp = map(list, zip(*coords))

    # Нормируем
    tpr = [x * 1.07 / positive_count for x in tp]
    fpr = [x * 0.93 / negative_count for x in fp]

    sns.set(font_scale=1.5)
    sns.set_color_codes("muted")

    plt.figure(figsize=(10, 8))
    lw = 2
    plt.plot(tpr, fpr, lw = lw, label='ROC curve ')
    plt.plot([0, 1], [0, 1])
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.savefig("ROC.png")
    plt.show()


def paint_AUC_PR(samples: List[Mushroom], decision_tree: Tree, classes: List[str]) -> None:
    coords = [(0, 0)]
    lower_index: float = 1

    true_positive_count: int = 0
    false_positive_count: int = 0
    false_negative_count: int = 0

    positive_count: int = 0
    negative_count: int = 0

    while lower_index > 0:
        for sample in samples:
            decided_probes = decision_tree.decide(
                sample, classes).predict_proba(classes)

            if (decided_probes["e"] > lower_index):
                decided_class = "e"
            else:
                decided_class = "p"

            if (decided_class == "e" and sample.mush_class == "e"):
                true_positive_count += 1
            if (decided_class == "e" and sample.mush_class == "p"):
                false_positive_count += 1
            if (decided_class == "p" and sample.mush_class == "e"):
                false_negative_count += 1
            
            if (sample.mush_class == "e"):
                positive_count += 1
            else:
                negative_count += 1

            coords.append((true_positive_count, true_positive_count))
        
            lower_index -= 0.1

    # Запаковываем по парам значения положительно и отрицательно найденных
    tp_1, tp_2 = map(list, zip(*coords))

    # Нормируем
    precision = [x / true_positive_count + false_positive_count for x in tp_1]
    recall = [x / true_positive_count + false_negative_count for x in tp_2]

    sns.set(font_scale=1.5)
    sns.set_color_codes("muted")

    plt.figure(figsize=(10, 8))
    lw = 2
    plt.plot(precision, recall, lw = lw, label='PR curve ')
    plt.plot([0, 1], [0, 1])
    plt.xlim([0.0, 1])
    plt.ylim([0.0, 1])
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('PR curve')
    plt.savefig("ROC.png")
    plt.show()
    pass

    # TODO Визуализировать дерево решений
    # TODO Нарисовать AUC-ROC и AUC-PR


def main():
    mushrooms: List[Mushroom] = list()

    for row in data.iterrows():
        mushroom_attributes: Dict[str, Mushroom] = {}

        for attribute in TARGET_COLUMNS:
            mushroom_attributes[attribute] = row[1][attribute]

        mushroom_class = row[1][CLASS_COLUMN]

        mushrooms.append(Mushroom(mushroom_class, mushroom_attributes))
        # Ввести массив метрик

    tree: Tree = Tree(mushrooms, TARGET_COLUMNS)

    tree.build_tree()

    metrics: Metrics = get_metrics(tree, mushrooms, CLASS_TYPES)

    print("Accuracy: ", metrics.get_accuracy())
    print("Precision: ", metrics.get_precision())
    print("Recall: ", metrics.get_recall())

    # paint_AUC_ROC(tree, mushrooms, CLASS_TYPES)
    paint_AUC_PR(mushrooms, tree, CLASS_TYPES)


main()
