import os

from typing import List, Dict, Tuple
from itertools import combinations, chain, product


max_level = 2
accumulated_features = {'numeric': [], 'discrete': []}

def _check_commutative(first: str, second: str, op:str):
    if (first in second) and (op in second):
        return True

def filter_binary_op(pair: Tuple[str,str], op: str) -> bool:
    ''' filter commutative operation like: (a, AddBinaryOp(a,b))
        if need to filter, return True, otherwise, return False'''
    if op not in ['AddBinaryOperator', 'MultiplyBinaryOperator']:
        if (op in ['SubtractBinaryOperator', 'DivisionBinaryOperator']) and (pair[0] == pair[1]):
            return True
        return False
    if _check_commutative(pair[0], pair[1], op) or _check_commutative(pair[1], pair[0], op):
        return True
    return False

def copy_features_and_add(features: Dict[str, List[str]], key: str, new_feature: str) -> Dict[str, List[str]]:
    '''copy of dictionary is not enough, it's not deep copy inner lists, thus we copy relevant list ourself'''
    copy = features.copy()
    copy[key] = features[key].copy()
    copy[key].append(new_feature)
    return copy

def count_operations(original_features: Dict[str, List[str]], unaryOperators: List[str], unaryNumToDiscreteOperators: List[str],
                     nonUnaryOperators: List[str], nonUnaryDiscreteToNumOperators: List[str], added_features: Dict[str, List[str]], level: int):
    count = 0
    if level == 0:
        return 1

    if len(added_features['numeric']) == 0:
        features = original_features
    else:
        features = added_features

    # for each numeric feature, apply unary operator -> adds new numeric column
    # for each numeric feature, apply unary discretize operator -> add new discrete column
    for feature in features['numeric']:
        for op in unaryOperators:
            updated = unaryOperators.copy()
            updated.remove(op)
            new_feature = f'{op}({feature})'
            if new_feature not in features['numeric']:
                added_features_copy = copy_features_and_add(added_features, 'numeric', new_feature)
                accumulated_features['numeric'].append(new_feature)
                count += count_operations(original_features, updated, unaryNumToDiscreteOperators, nonUnaryOperators,
                                          nonUnaryDiscreteToNumOperators, added_features_copy, level - 1)

        for op in unaryNumToDiscreteOperators:
            updated = unaryNumToDiscreteOperators.copy()
            # updated.remove(op)
            new_feature = f'{op}({feature})'
            if new_feature not in features['discrete']:
                added_features_copy = copy_features_and_add(added_features, 'discrete', new_feature)
                accumulated_features['discrete'].append(new_feature)
                count += count_operations(original_features, unaryOperators, updated, nonUnaryOperators,
                                          nonUnaryDiscreteToNumOperators, added_features_copy, level - 1)

    # for each pair of 2 numeric features, apply binary operator -> add new numeric column
    possible_pairs = list(product(original_features['numeric'], features['numeric']))
    for op in nonUnaryOperators:
        updated = nonUnaryOperators.copy()
        # updated.remove(op)
        for pair in possible_pairs:
            if filter_binary_op(pair, op): continue
            new_feature = '{}({},{})'.format(op, pair[0], pair[1])
            if new_feature not in features['numeric']:
                added_features_copy = copy_features_and_add(added_features, 'numeric', new_feature)
                accumulated_features['numeric'].append(new_feature)
                count += count_operations(original_features, unaryOperators, unaryNumToDiscreteOperators, updated,
                                          nonUnaryDiscreteToNumOperators, added_features_copy, level - 1)

    # for each subset of discrete features (as source) and a numeric feature (as target),
    # apply a high-order operator -> add new numeric column
    possible_subsets = chain.from_iterable([list(combinations(features['discrete'], i)) for i in range(1, len(features['discrete']) + 1)])
    for tup in possible_subsets:
        for op in nonUnaryDiscreteToNumOperators:
            updated = nonUnaryDiscreteToNumOperators.copy()
            # updated.remove(op)
            new_feature = '{}({})'.format(op, ', '.join(tup))
            added_features_copy = copy_features_and_add(added_features, 'numeric', new_feature)
            accumulated_features['numeric'].append(new_feature)
            count += count_operations(original_features, unaryOperators, unaryNumToDiscreteOperators,
                                      nonUnaryOperators, updated, added_features_copy, level - 1)

    return count

def name_features(features: Dict[str, int]):
    '''
    :param features: {'numeric': 3, 'discrete': 1}
    :return: output: {'numeric': ['a','b', 'c'], 'discrete': ['A']}
    give features alphanumeric name
    '''
    features = {'numeric': [chr(ord('a') + i) for i in range(features['numeric'])],
                'discrete': [chr(ord('A') + i) for i in range(features['discrete'])]}
    return features

def discretize_before_generation(features: List[str], unaryNumToDiscreteOperators):
    new_feat = []
    for op in unaryNumToDiscreteOperators:
        for feat in features:
            new_feat.append(f'{op}({feat})')
    return new_feat

def main():
    unaryOperators = 'StandardScoreUnaryOperator'  # ,HourOfDayUnaryOperator,DayOfWeekUnaryOperator,IsWeekendUnaryOperator'
    unaryNumToDiscreteOperators = 'EqualRangeDiscretizerUnaryOperator'
    nonUnaryOperators = 'AddBinaryOperator, DivisionBinaryOperator, MultiplyBinaryOperator,SubtractBinaryOperator'
    nonUnaryDiscreteToNumOperators = 'GroupByThenAvg, GroupByThenStdev, GroupByThenCount, GroupByThenMax, GroupByThenMin'
    unaryOperators = unaryOperators.split(',')
    unaryNumToDiscreteOperators = unaryNumToDiscreteOperators.split(',')
    nonUnaryOperators = nonUnaryOperators.replace(' ', '').split(',')
    nonUnaryDiscreteToNumOperators = nonUnaryDiscreteToNumOperators.replace(' ', '').split(',')

    features = {'numeric': 2, 'discrete': 0}
    features = name_features(features)
    features['discrete'] += discretize_before_generation(features['numeric'], unaryNumToDiscreteOperators)

    count = count_operations(features, unaryOperators, unaryNumToDiscreteOperators, nonUnaryOperators,
                             nonUnaryDiscreteToNumOperators, {'numeric': [], 'discrete': []}, max_level)

    print('Results:')
    print(f'{len(accumulated_features["numeric"])} numeric features')
    print(f'{len(accumulated_features["discrete"])} discrete features')
    print('Numeric:' + os.linesep.join(accumulated_features['numeric']))
    print('Discrete:' + os.linesep.join(accumulated_features['discrete']))

if __name__=='__main__':
    main()