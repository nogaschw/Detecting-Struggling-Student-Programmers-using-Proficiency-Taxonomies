import numpy as np
import pandas as pd

class Falcon:
    def __init__(self, also50unti80=True):
        self.name = 'falcon'
        self.embedded = ['falcon_to_model_output']
        path = "/home/nogaschw/Codeworkout/df_taxonomy_falcon_new_label.pkl"
        df = pd.read_pickle(path)
        df.fillna(0, inplace=True)
        df['prev_code'] = df['prev_code'].apply(lambda x: [list(set(i)) for i in x])
        if not also50unti80:
            self.df = self._arrange(df)
        else: 
            self.df = pd.concat([self._arrange(df), self._arrange(df, 50, 80)], ignore_index=True)
    
    def _arrange(self, df1, st_point=0, end_point=30):
        df1 = df1[df1['prev_question'].apply(lambda x: len(x) > end_point and len(x) <= end_point + 20)]
        df1['prev_code'] = df1['prev_code'].apply(lambda x: x[st_point:end_point])
        df1['prev_question'] = df1['prev_question'].apply(lambda x: x[st_point:end_point])
        df1['prev_label'] = df1['prev_label'].apply(lambda x: x[st_point:end_point])
        df1 = df1[df1['student_id'].map(df1['student_id'].value_counts()) == 20]
        return df1

class Codeworkout:
    def __init__(self):
        self.name = 'codeworkout'
        self.embedded = ['codeworkout_to_model_output']
        path = "/home/nogaschw/Codeworkout/df_codeworkout_TL_Tax.pkl"
        df = pd.read_pickle(path)
        df = self._order_features(df)
        df['prev_code'] = df['prev_code'].apply(lambda x: [list(set(i)) for i in x])
        df = df.apply(self._sort_arrays_by_dates, axis=1)
        path = "/home/nogaschw/Codeworkout/LinkTables/questions.xlsx"
        questions_df = pd.read_excel(path)
        questions_df = self._order_features(questions_df)
        questions_df = questions_df.drop(columns=['Unnamed: 21', 'Unnamed: 22', 'Unnamed: 23'])
        cols_to_merge = ['IfElse', 'Loops', 'MathOperations', 'LogicOperators', 'StringOperations', 'List', 'FileOperations', 'Functions', 'Dictionary', 'Tuple']
        questions_df['comp_cons'] = questions_df[cols_to_merge].apply(lambda row: row.values.tolist(), axis=1)
        questions_df = questions_df.drop(columns=cols_to_merge)
        Comp_cons_dict = questions_df.set_index('Requirement')['comp_cons'].to_dict()
        df['prev_comp_cons'] = df['prev_question'].apply(lambda x: [Comp_cons_dict[i] for i in x])
        self.df = df

    def _order_features(self, df):
        df.fillna(0, inplace=True)
        df['IfElse'] = df[['If/Else', 'NestedIf']].max(axis=1)
        df['Loops'] = df[['While', 'For', 'NestedFor']].max(axis=1)
        df['MathOperations'] = df[['Math+-*/', 'Math%']].max(axis=1)
        df['LogicOperators'] = df[['LogicAndNotOr', 'LogicCompareNum', 'LogicBoolean']].max(axis=1)
        df['StringOperations'] = df[["StringFormat", "StringConcat", "StringIndex", "StringLen", "StringEqual", "CharEqual"]].max(axis=1)
        df["List"]= df["ArrayIndex"]
        df['FileOperations'] = 0
        df["Functions"] = df["DefFunction"]
        df['Dictionary'] = 0
        df['Tuple'] = 0
        df = df.drop(columns=['If/Else', 'NestedIf', 'StringConcat', 'StringIndex', 'StringFormat',"StringLen", "StringEqual", 'ArrayIndex', 'DefFunction',
                            'While', 'For', 'NestedFor', 'Math+-*/', 'Math%', 'LogicAndNotOr', 'LogicCompareNum', 'LogicBoolean', 'CharEqual'])
        return df

    def _sort_arrays_by_dates(self, row):
        # Get sorting indices based on dates
        sorted_indices = np.argsort(row['prev_start_time'])
        row['prev_start_time'] = [row['prev_start_time'][i] for i in sorted_indices]
        row['prev_label'] = [row['prev_label'][i] for i in sorted_indices]
        row['prev_code'] = [row['prev_code'][i] for i in sorted_indices]
        row['prev_question'] = [row['prev_question'][i] for i in sorted_indices]
        return row
    
class Both:
    def __init__(self, also50unti80=True):
        self.name = 'both'
        self.falcon_df = Falcon(also50unti80)
        self.codeworkout_df = Codeworkout()
        self.embedded = [self.falcon_df.embedded, self.codeworkout_df.embedded]
        self.df = pd.concat([self.falcon_df, self.codeworkout_df], axis=0, ignore_index=True, join='outer')