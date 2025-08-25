class Config:
    def __init__(self):
        self.codeworkout_questions_path = '/home/nogaschw/Codeworkout/OriginalData/codeworkout/questions.xlsx'
        self.falconcode_questions_path = '/home/nogaschw/Codeworkout/OriginalData/falcon/cleaned_questions.csv'
        self.path_saved_codeworkout = '/home/nogaschw/Codeworkout/Dataset/codeworkout.pkl'
        self.path_saved_falcon = '/home/nogaschw/Codeworkout/Dataset/falcon.pkl'
        self.text_model_name = 'sentence-transformers/all-MiniLM-L6-v2' #'google-bert/bert-base-cased' 
        self.lr = 0.0001
        self.batch_size = 32
        self.max_len_code = 768
        self.padding_size_code = 100
        self.dataset = 1 # 0 for codeworkout or 1 for falcon
        name = ['CW', 'FC', 'Both'][self.dataset]
        self.weight_binary = 0.5
        self.config_name = f'{name}Allresults'
        self.submission_hl = 512 
        self.snapshots_hl = 512 if self.padding_size_code != 1 else self.max_len_code
        self.dropout_rate = 0.3
        self.run_model = 'PTM' # for Ablation change the name
        self.number_coding_tasks = 30
        self.gemma_path = f'/home/nogaschw/Codeworkout/Thesis/Gemma/{name}snfirst.csv' \
            if self.dataset != 2 else [f'/home/nogaschw/Codeworkout/Thesis/Gemma/CWsnfirst.csv',
            f'/home/nogaschw/Codeworkout/Thesis/Gemma/FCsnfirst.csv']
        self.epoch = 15