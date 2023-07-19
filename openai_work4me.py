import os
import openai
import tqdm
import json
import pickle
openai.organization = "org-60G0Vagv1LXp72arNooqJxzm"
openai.api_key = "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"
# openai.Model.list()
dataset_path = '../../fin_datasets'
dataset_name = 'ccks2021金融事件因果关系抽取/ccks_task2_train.txt'
data_path = os.path.join(dataset_path,dataset_name)

data_list = []
with open(data_path, "r") as f:
    for line in f:
        example = json.loads(line)
        data_list.append(example)
        
            
# import pickle
# with open('your_data.pkl','rb') as f:
#     news = pickle.loads(f.read())

instruction = """\n---
从上述文本中抽取出所有包含的因果关系事件，给出原因事件类型和结果事件类型，类型不会为空，并找出每个事件类型下的影响地域、产品、行业等是三个事件要素，类型可以返回空。
最终输出{"reason_type":"xxx","reason_region":[],"reason_product":[],"reason_industry":["xxx"],"result_type":"xxx","result_region":[],"result_product":["xxx"],"result_industry":[]}的json数据。
定义的原因事件类型有['供给减少','市场价格下降','市场价格提升','供给增加','需求减少','需求增加','限产','猪瘟','其他贸易摩擦','干旱','其他自然灾害','负向影响','销量（消费）减少','销量（消费）增加','运营成本提升','禽流感','正向影响','洪涝','台风','进口下降','进口增加','出口下降','运营成本下降','地震','出口增加','其他畜牧疫情','产品利润下降','寒潮','产品利润增加','对华加征关税','对华反倾销','霜冻','对他国反倾销','其他或不明确','猪口蹄疫','滞销','冰雹','山洪']，结果事件类型有['市场价格提升','市场价格下降','供给减少','产品利润下降','需求减少','产品利润增加','销量（消费）减少','需求增加','负向影响','运营成本提升','供给增加','销量（消费）增加','正向影响','进口下降','运营成本下降','出口下降','进口增加','出口增加','其他或不明确']。"""


import tiktoken
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
def get_num_tokens(text):
    return len(encoding.encode(text))


contents = [t['text'] + instruction for t in data_list]


def get_openai_res(content):
    try:
        completion = openai.ChatCompletion.create(
        #   model="gpt-3.5-turbo",
        model="gpt-4",
          messages=[
            {"role": "user", "content": content}
          ]
        )
        msg = completion.choices[0].message['content']
    except:
        msg = ''
    return [content,msg]


import concurrent.futures
results = []

with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    futures = {executor.submit(get_openai_res, content) for content in contents}

    for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
        results.append(future.result())
        # 这里，每当有任务完成，就会打印一次进度
        print(f"Processed {i}/{len(contents)} contents.")
        if i % 50 == 0:
            with open('sentiment_comp_qaie_pairs_gpt4.pkl','wb') as f:
                pickle.dump(results,f)
                print('saved',i)
            
len(results)
with open('sentiment_comp_qaie_pairs_gpt4.pkl','wb') as f:
    pickle.dump(results,f)