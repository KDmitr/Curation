import ast
import pandas as pd
import json

def has_curly_brace(text):
    return '{' in text or '}' in text

def extract_json_objects(text):
    json_objects = []
    stack = []
    start_index = -1
    
    for i, char in enumerate(text):
        if char == '{':
            if not stack:
                start_index = i  
            stack.append(char)
        elif char == '}':
            if stack:
                stack.pop()
                if not stack: 
                    json_str = text[start_index:i+1]
                    try:
                        obj = json.loads(json_str)
                        json_objects.append(obj)
                    except json.JSONDecodeError:
                        
                        pass
    
    return json_objects

def extract_json(input_string):
    start_pos = input_string.find('[')  
    if start_pos == -1:
        return None  

    nesting_level = 0
    for i in range(start_pos, len(input_string)):
        if input_string[i] == '[':
            nesting_level += 1
        elif input_string[i] == ']':
            nesting_level -= 1
        
        if nesting_level == 0:  
            end_pos = i + 1  
            json_string = input_string[start_pos:end_pos]
            json_string.replace('\n','')
            
            try: 
                data_ = ast.literal_eval(json_string)
                return data_
            except:
                continue
    return None  

class GEN():
    def __init__(self, gen_client, gen_model_nm, real_data, cols,y_col, num_cols, gen_temperature=0.5, real_samples_num=2) -> None:
        self.gen_client = gen_client
        self.gen_model_nm = gen_model_nm
        self.params = None
        self.cols = cols
        self.y_col = y_col
        self.num_cols = num_cols
        real_data.reset_index(inplace=True, drop=True)
        self.real_data = real_data
        
        self.tobe_refined = ""

        pred_true_df = pd.get_dummies(real_data[cols])
        self.pred_Xcols = list(pred_true_df.columns)
        self.res = {} 
        self.res_df = [] 
        self.sample = None 
        self.df_syn=None
        self.res_df = []
        self.model = None
        self.dict_template = {}
        self.ress=None
        
        for col in cols:
            self.dict_template[col] = []
        
        temp = []
        for i in range(real_samples_num):
            temptemp = '{' + f'sample {i}' + '}'
            temp.append(temptemp)
        self.response_template = str(temp)

        self.real_samples_num = real_samples_num

        self.gen_temperature = gen_temperature
    

    def instruction(self, sample):
        prompt_sys = """
   The ultimate goal is to produce accurate and convincing synthetic
    data given the user provided samples."""

        prompt_user = f"""Here are examples from real data: {sample}\n"""
        prompt_user += f"Generate {self.real_samples_num} synthetic sample mimics the provided samples. DO NOT COPY the sample. The response should be formatted SCTRICTLY as a list in JSON format, which is suitable for direct use in data processing scripts such as conversion to a DataFrame in Python. No additional text or numbers should precede the JSON data."
        return prompt_sys, prompt_user
    
    def row2dict(self, rows):
        rows.reset_index(inplace=True, drop=True)
        res = []
        for i in range(len(rows)):
            example_data = {}
            row = rows.iloc[i, :]
            for x in self.cols:
                if x in self.num_cols:
                    example_data[x] = round(row[x], 4)
                else:
                    example_data[x] = row[x]
            res.append(example_data)
        return str(res)
    

    def gen(self, batch_size, i=0, epoch=0):
        res = []
        j = i
        if j + batch_size <= len(self.real_data):
            self.sample = self.real_data.loc[j:(j+batch_size), self.cols].copy()
        else:
            self.sample = self.real_data.loc[j:, self.cols].copy()
        
        while j < i + batch_size:
            sampled_rows = self.real_data.loc[j : (j+self.real_samples_num-1), self.cols].copy()

            sample = self.row2dict(sampled_rows)
            sys_info, user_info = self.instruction(sample)
            print(f'sys_info: {sys_info}')
            print(f'user_info: {user_info}')
            resp_temp = self.gen_client.chat.completions.create(
                    model=self.gen_model_nm, 
                    messages=[
                        {"role": "system", "content": sys_info },
                        {"role": "user", "content": user_info}
                    ],
                    temperature=self.gen_temperature,
                    n = 1,max_tokens=3000, extra_headers={"X-Title": "Anonymized datasets"}
            )
           
            if resp_temp.choices[0].message.content is not None:
                
                print(type(resp_temp.choices[0].message.content))
            
            res.append(extract_json_objects(resp_temp.choices[0].message.content))

            j = j + self.real_samples_num
        index = str(epoch)+'-'+str(i)
        self.res = res


    def process_response(self, resp_lst):
        res = {}
        for key, val in self.dict_template.items():
            res[key] = []
        self.json_err = 0
        self.no_group_err = 0
        self.var_key_err = 0
        self.dict_error = 0
        print(resp_lst)
        json_temp = extract_json(resp_lst)
           
        return pd.DataFrame(json_temp)
    
    
    def isValid(self, s):
        #s=str(s)
        stack=[]
        match={'{':'}'}
        for i in s:
            if i in ['{']:
                stack.append(i)
            if i in ['}']:
                stack.pop()
        return stack==[]
    
    
    def run(self, name=''):
        i = 0
        e = 0
        self.gen(len(self.real_data), i=i, epoch=e)
        
        records = []
        for sublist in self.res:
            if sublist: 
                records.extend(sublist)

        df_temp = pd.DataFrame(records, columns=list(self.real_data.columns))

        df_temp['real_identifier'] = [0] * len(df_temp)
        df_true = self.real_data
        df_true['real_identifier'] = [1] * len(df_true)
        df_comb = pd.concat([df_temp, df_true])
        df_comb.to_csv(name +'.csv')
        self.df_syn=df_temp
        return df_comb
