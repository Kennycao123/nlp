### 1.功能

新闻文本进行分类

id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 11, 12, 13, 14]

item = ['民生','文化','娱乐','体育','财经','房产','汽车',
'教育','科技','军事','旅游 ','国际','证券','农业','电竞']

目前基于头条数据集（４０万）进行训练对文本进行分类

### 2.目录结构

├── Readme.md                   　　　// help
├── data                        　　　// 数据
│   ├── text-classfication-dataset　　// 数据集
│   │   ├── abstract_news.csv 　　　　// 抽取的数据(标签100-116)
│   │   ├── classf_news.csv   　　　　// 抽取的数据(转换标签0-14)
│   │   ├── classf_news_add.csv　 　　// 数据集增加，补充部分数据少的分类
│   │   ├── get_data.py　　　　　 　　// 抽取数据
│   │   ├── readme.md　　　　　　 　　// 数据介绍
│   │   ├── t.py     　　　　　　 　　// 数据集标签转换和数据分割
│   │   ├── t2.py    　　　　　　 　　// 从另一个数据集抽取数据
│   │   ├── t3.py    　　　　　　 　　// 增添数据
│   │   ├── text_agriculture_list.csv // 农业数据
│   │   ├── text_edu_list.csv  　 　　// 教育数据
│   │   ├── text_house_list.csv　 　　// 房产数据
│   │   ├── text_society_list.csv 　　// 社会数据
│   │   ├── text_stock_list.csv　 　　// 股票数据
│   │   ├── text_story_list.csv　 　　// 故事文化数据
│   │   ├── toutiao_cat_data.txt  　　// 原始数据文本
│   │   ├── train_text.csv 　　　 　　// 训练数据
│   │   └── val_text.csv 　　　　 　　// 验证数据
│   ├── abstract_news.csv             // 新闻抽取的摘要
│   ├── get_abstract.py 　　　　　　　// 抽取摘要
│   ├── public_opinion.json           // 原始数据
│   ├── remove_same.py　　　　　　　　// 去除相同文本
│   └── public_opinion_news.csv       // 原始数据抽取新闻主体
├── models                      　　　// 保存权重文件
├── __init__.py
├── data　　　　　　　　　　　　　　　// 数据处理文件(向量化)
├── export.py　　       　　　　　　　// 生成预测后的数据集
├── model.py　                        // 训练和测试bert模型
├── t1.py                             // docker调用
├── test.py　                         // 测试模型效果
├── train.py　                        // 训练模型权重
└── utils.py　                        // 功能函数

### 3.验证精度

验证loss: 0.2150, accuracy: 94.3064%, auc: 0.9963

### 4.存在的额问题

类别多，数据分类模糊，民生和文化界限不清晰，文本均衡后精度依然不高，
训练文本为短文本，信息量少，基于语义理解相对较差，个别字的权重影响较大

希望能够以长文本的数据集进行训练提高精度


### 5.环境配置

transformers==2.3.0