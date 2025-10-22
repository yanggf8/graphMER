import csv
from typing import Dict, List, Any, Optional, Union, Callable
from collections import defaultdict
import json

class Series:
    def __init__(self, data: List[Any], index: List[Any] = None):
        self.data = data
        self.index = index or list(range(len(data)))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, key):
        if isinstance(key, int):
            return self.data[key]
        elif key in self.index:
            idx = self.index.index(key)
            return self.data[idx]
        raise KeyError(key)
    
    def sum(self):
        return sum(x for x in self.data if isinstance(x, (int, float)))
    
    def mean(self):
        numeric_data = [x for x in self.data if isinstance(x, (int, float))]
        return sum(numeric_data) / len(numeric_data) if numeric_data else 0
    
    def unique(self):
        return list(set(self.data))
    
    def value_counts(self):
        counts = defaultdict(int)
        for item in self.data:
            counts[item] += 1
        return dict(counts)

class DataFrame:
    def __init__(self, data: Dict[str, List[Any]] = None):
        self.data = data or {}
        self.columns = list(self.data.keys())
        self.index = list(range(len(next(iter(self.data.values()), []))))
    
    def __getitem__(self, key: Union[str, List[str]]):
        if isinstance(key, str):
            return Series(self.data[key], self.index)
        elif isinstance(key, list):
            subset = {col: self.data[col] for col in key if col in self.data}
            return DataFrame(subset)
        raise KeyError(key)
    
    def __setitem__(self, key: str, value: Union[List[Any], Any]):
        if isinstance(value, list):
            self.data[key] = value
        else:
            self.data[key] = [value] * len(self.index)
        
        if key not in self.columns:
            self.columns.append(key)
    
    @property
    def shape(self):
        rows = len(self.index)
        cols = len(self.columns)
        return (rows, cols)
    
    def head(self, n: int = 5):
        subset = {}
        for col in self.columns:
            subset[col] = self.data[col][:n]
        return DataFrame(subset)
    
    def tail(self, n: int = 5):
        subset = {}
        for col in self.columns:
            subset[col] = self.data[col][-n:]
        return DataFrame(subset)
    
    def describe(self):
        stats = {}
        for col in self.columns:
            series = Series(self.data[col])
            numeric_data = [x for x in series.data if isinstance(x, (int, float))]
            if numeric_data:
                stats[col] = {
                    'count': len(numeric_data),
                    'mean': sum(numeric_data) / len(numeric_data),
                    'min': min(numeric_data),
                    'max': max(numeric_data)
                }
        return stats
    
    def groupby(self, column: str):
        return GroupBy(self, column)
    
    def sort_values(self, by: str, ascending: bool = True):
        if by not in self.columns:
            raise KeyError(f"Column '{by}' not found")
        
        # Create list of (value, index) pairs
        indexed_values = [(self.data[by][i], i) for i in range(len(self.index))]
        indexed_values.sort(key=lambda x: x[0], reverse=not ascending)
        
        # Reorder all columns based on sorted indices
        sorted_data = {}
        for col in self.columns:
            sorted_data[col] = [self.data[col][idx] for _, idx in indexed_values]
        
        return DataFrame(sorted_data)
    
    def drop(self, columns: Union[str, List[str]]):
        if isinstance(columns, str):
            columns = [columns]
        
        new_data = {col: self.data[col] for col in self.columns if col not in columns}
        return DataFrame(new_data)
    
    def fillna(self, value: Any):
        new_data = {}
        for col in self.columns:
            new_data[col] = [value if x is None else x for x in self.data[col]]
        return DataFrame(new_data)
    
    def apply(self, func: Callable, axis: int = 0):
        if axis == 0:  # Apply to columns
            result = {}
            for col in self.columns:
                result[col] = func(Series(self.data[col]))
            return result
        else:  # Apply to rows
            results = []
            for i in range(len(self.index)):
                row_data = {col: self.data[col][i] for col in self.columns}
                results.append(func(row_data))
            return Series(results)
    
    def to_csv(self, filename: str):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.columns)
            for i in range(len(self.index)):
                row = [self.data[col][i] for col in self.columns]
                writer.writerow(row)
    
    def to_json(self, filename: str = None):
        json_data = []
        for i in range(len(self.index)):
            row = {col: self.data[col][i] for col in self.columns}
            json_data.append(row)
        
        if filename:
            with open(filename, 'w') as f:
                json.dump(json_data, f, indent=2)
        return json_data

class GroupBy:
    def __init__(self, df: DataFrame, column: str):
        self.df = df
        self.column = column
        self.groups = self._create_groups()
    
    def _create_groups(self):
        groups = defaultdict(list)
        for i, value in enumerate(self.df.data[self.column]):
            groups[value].append(i)
        return dict(groups)
    
    def sum(self):
        result = {}
        for group_value, indices in self.groups.items():
            group_sums = {}
            for col in self.df.columns:
                if col != self.column:
                    values = [self.df.data[col][i] for i in indices]
                    numeric_values = [v for v in values if isinstance(v, (int, float))]
                    group_sums[col] = sum(numeric_values) if numeric_values else 0
            result[group_value] = group_sums
        return result
    
    def count(self):
        return {group_value: len(indices) for group_value, indices in self.groups.items()}
    
    def mean(self):
        result = {}
        for group_value, indices in self.groups.items():
            group_means = {}
            for col in self.df.columns:
                if col != self.column:
                    values = [self.df.data[col][i] for i in indices]
                    numeric_values = [v for v in values if isinstance(v, (int, float))]
                    if numeric_values:
                        group_means[col] = sum(numeric_values) / len(numeric_values)
                    else:
                        group_means[col] = 0
            result[group_value] = group_means
        return result

def read_csv(filename: str) -> DataFrame:
    data = defaultdict(list)
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for key, value in row.items():
                # Try to convert to numeric
                try:
                    value = float(value)
                    if value.is_integer():
                        value = int(value)
                except ValueError:
                    pass
                data[key].append(value)
    return DataFrame(dict(data))

def read_json(filename: str) -> DataFrame:
    with open(filename, 'r') as f:
        json_data = json.load(f)
    
    if isinstance(json_data, list):
        data = defaultdict(list)
        for row in json_data:
            for key, value in row.items():
                data[key].append(value)
        return DataFrame(dict(data))
    else:
        raise ValueError("JSON must be a list of objects")
