class Config:
    def __init__(self):
        self.codeworkout_folder = 'OriginalData/codeworkout_competition'
        self.codeworkout_courses = ['Spring', 'Train_Fall', 'Test_Fall']
        self.falconcode_folder = 'OriginalData/falcon'
        self.path_tosave_codeworkout = 'Dataset/codeworkout_competition.pkl'
        self.path_tosave_falcon = 'Dataset/falcon.pkl'
        self.code_model_name = 'microsoft/codebert-base'
        self.save_code_embedding = 'Dataset/falcon_to_model_output.pkl'
        self.max_model_len = 512