import os
import re
import time
import random
import requests
import jieba
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from wordcloud import WordCloud
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from fake_useragent import UserAgent
from snownlp import SnowNLP

# ================ 全局字体初始化 ================
# 必须在其他导入之前设置
mpl.use('Agg')  # 解决无GUI环境问题

# 配置系统字体（Windows/Mac/Linux自动适配）
try:
    if os.name == 'nt':  # Windows系统
        mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    else:  # Mac/Linux系统
        mpl.rcParams['font.sans-serif'] = ['PingFang HK', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei']

    mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.rcParams['font.size'] = 12  # 全局字体大小

    # 验证字体配置
    test_fig, test_ax = plt.subplots()
    test_ax.set_title("中文测试")
    test_fig.savefig(os.path.join(os.getcwd(), 'font_test.png'))
    plt.close(test_fig)
    print("✅ 系统字体配置验证通过")
except Exception as e:
    print("❌ 字体配置失败:", str(e))
    print("请执行以下解决方案：")
    print("1. Windows系统：安装[微软雅黑](https://learn.microsoft.com/zh-cn/typography/font-list/microsoft-yahei)")
    print("2. Mac/Linux系统：执行安装命令：sudo apt install fonts-noto-cjk")
    exit(1)

# ================== 配置 ==================
CONFIG = {
    # 多电影配置（豆瓣ID: 电影名称）
    'movies': {
        '30181250': '封神第二部：战火西岐',
        '36282639': '唐探1900',
        '34780991': '哪吒之魔童闹海',
        '36289423': '射雕英雄传：侠之大者',
        '35295960': '蛟龙行动',
        '36970301':'熊出没·重启未来',
    },
    'output_dir': './reports',  # 输出目录
    'page_limit': 5,  # 每部电影抓取页数
    'max_workers': 5,  # 并发线程数
    'proxy_pool': [  # 代理IP池
        # 'http://ip1:port',
        # 'http://ip2:port'
    ],
    'filterRoleNames':True,
    'font_path':'./font/NotoSansCJKMedium.otf',  # 字体
    'filterText':'./filterText.txt',
    'stopwords': './stopwords.txt',  # 停用词文件路径
    'sentiment_threshold': (0.4, 0.6)  # 情感阈值(负面, 中性)
}


# =============================================

class MovieAnalyzer:
    def __init__(self):
        self.ua = UserAgent()
        os.makedirs(CONFIG['output_dir'], exist_ok=True)

        # 初始化分词器
        jieba.load_userdict('./userdict.txt')  # 自定义词典

        # 加载停用词
        with open(CONFIG['stopwords'], 'r', encoding='utf-8') as f:
            self.stopwords = set(f.read().splitlines())

    def get_headers(self):
        """生成动态请求头"""
        return {
            'User-Agent': self.ua.random,
            'Referer': 'https://movie.douban.com/'
        }

    def get_proxy(self):
        """随机获取代理IP"""
        return random.choice(CONFIG['proxy_pool']) if CONFIG['proxy_pool'] else None

    def fetch_data(self, movie_id):
        """多线程安全的数据抓取"""
        all_comments = []
        character_blacklist = []

        try:
            # 获取演员表
            url = f'https://movie.douban.com/subject/{movie_id}/celebrities'
            resp = requests.get(url, headers=self.get_headers(),
                                proxies={'http': self.get_proxy()}, timeout=15)
            soup = BeautifulSoup(resp.text, 'html.parser')
            if CONFIG['filterRoleNames']:
                character_blacklist = [li.find('span', class_='name').text
                                   for li in soup.select('li.celebrity')[:8]]


            # 获取短评
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = []
                for page in range(CONFIG['page_limit']):
                    futures.append(
                        executor.submit(self._fetch_page_comments,
                                        movie_id, page)
                    )
                    time.sleep(random.uniform(0.5, 1.5))
                for future in futures:
                    all_comments.extend(future.result())

        except Exception as e:
            print(f'电影{movie_id}数据获取异常: {str(e)}')

        return {
            'comments': all_comments,
            'characters': character_blacklist
        }

    def _fetch_page_comments(self, movie_id, page):
        """单页评论抓取"""
        try:
            url = f'https://movie.douban.com/subject/{movie_id}/comments?start={page * 20}'
            print('url:',url)
            resp = requests.get(url, headers=self.get_headers(),
                                proxies={'http': self.get_proxy()}, timeout=10)
            soup = BeautifulSoup(resp.text, 'html.parser')
            comments = [self._clean_text(span.get_text())
                        for span in soup.select('span.short')]
            time.sleep(random.uniform(1, 3))
            print('获取第',page + 1,'页评论成功')
            return comments
        except Exception as e:
            print('获取第',page + 1 ,'页评论出错:', e)
            return []

    def _clean_text(self, text):
        """高级文本清洗"""
        text = re.sub(r'<[^>]+>', '', text)  # HTML标签
        text = re.sub(r'@\w+\s?', '', text)  # 去除@提及
        text = re.sub(r'【.*?】', '', text)  # 去除括号内容
        text = re.sub(r'[^\w\u4e00-\u9fff]', ' ', text)  # 保留中文和基本字符
        return text.strip()

    def analyze_movie(self, movie_id, movie_name):
        """核心分析流程"""
        print(f'🎬 正在分析《{movie_name}》...')
        data = self.fetch_data(movie_id)

        if not data['comments']:
            print(f'⚠️ 《{movie_name}》无有效评论')
            return

        # 情感分析与文本处理
        sentiment_results = []
        words = []
        characters_arr = []
        filter_text = []
        for ch in data['characters']:
            split_string = ch.split()
            characters_arr.extend(split_string)
        with open(CONFIG['filterText'], 'r', encoding='utf-8') as f:
            filter_text = set(f.read().splitlines())
        blacklist = set(characters_arr + list(filter_text))
        print("blacklist", blacklist)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for comment in data['comments']:
                futures.append(executor.submit(self._process_comment, comment, blacklist))

            for future in futures:
                result = future.result()
                if result:
                    words.extend(result['words'])
                    sentiment_results.append(result['sentiment'])

        # 生成分析报告
        self._generate_wordcloud(words, movie_name)
        self._generate_sentiment_chart(sentiment_results, movie_name)
        self._generate_full_report(words, sentiment_results, movie_name)

    def _process_comment(self, comment, blacklist):
        """处理单条评论（包含情感分析）"""
        try:
            # 情感分析
            s = SnowNLP(comment)
            sentiment = s.sentiments

            # 文本处理
            seg = jieba.lcut(comment)
            filtered_words = [w for w in seg if len(w) > 1
                              and w not in self.stopwords
                              and w not in blacklist]

            return {
                'words': filtered_words,
                'sentiment': sentiment
            }
        except:
            return None

    def _generate_wordcloud(self, words, movie_name):
        """生成高级词云"""
        freq = Counter(words)

        wc = WordCloud(
            font_path=CONFIG['font_path'],
            width=1600,
            height=1200,
            background_color='white',
            colormap='tab20',
            max_words=200,
            contour_width=1,
            contour_color='steelblue'
        ).generate_from_frequencies(freq)

        plt.figure(figsize=(20, 15))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(os.path.join(CONFIG['output_dir'],
                                 f'{movie_name}_词云.png'),
                    bbox_inches='tight', dpi=300)
        plt.close()

    def _generate_sentiment_chart(self, sentiments, movie_name):
        """生成情感分布饼图"""
        low, high = CONFIG['sentiment_threshold']
        counts = {
            '负面': sum(1 for s in sentiments if s < low),
            '中性': sum(1 for s in sentiments if low <= s <= high),
            '正面': sum(1 for s in sentiments if s > high)
        }

        plt.figure(figsize=(10, 10))
        plt.pie(
            counts.values(),
            labels=counts.keys(),
            autopct='%1.1f%%',
            colors=['#ff9999', '#66b3ff', '#99ff99'],
            startangle=90
        )
        plt.title(f'《{movie_name}》评论情感分布', fontsize=14)
        plt.savefig(os.path.join(CONFIG['output_dir'],
                                 f'{movie_name}_情感分布.png'),
                    bbox_inches='tight', dpi=150)
        plt.close()

    def _generate_full_report(self, words, sentiments, movie_name):
        """生成完整分析报告"""
        # 情感数据
        df_sentiment = pd.DataFrame({
            '情感得分': sentiments,
            '情感分类': ['正面' if s > CONFIG['sentiment_threshold'][1] else
                         '中性' if s >= CONFIG['sentiment_threshold'][0] else
                         '负面' for s in sentiments]
        })

        # 关键词数据
        freq = Counter(words)
        df_keywords = pd.DataFrame(freq.most_common(50),
                                   columns=['关键词', '频次'])

        # 保存Excel
        with pd.ExcelWriter(os.path.join(CONFIG['output_dir'],
                                         f'{movie_name}_分析报告.xlsx')) as writer:
            df_sentiment.to_excel(writer, sheet_name='情感分析', index=False)
            df_keywords.to_excel(writer, sheet_name='关键词分析', index=False)

            # 添加统计数据
            stats = pd.DataFrame({
                '指标': ['总评论数', '平均情感得分', '正面率', '负面率'],
                '数值': [
                    len(sentiments),
                    sum(sentiments) / len(sentiments),
                    sum(1 for s in sentiments if s > CONFIG['sentiment_threshold'][1]) / len(sentiments),
                    sum(1 for s in sentiments if s < CONFIG['sentiment_threshold'][0]) / len(sentiments)
                ]
            })
            stats.to_excel(writer, sheet_name='统计概览', index=False)

        # 生成趋势图
        plt.figure(figsize=(12, 6))
        df_keywords.head(15).plot.bar(x='关键词', y='频次', legend=False)
        plt.title(f'《{movie_name}》高频关键词TOP15')
        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG['output_dir'],
                                 f'{movie_name}_趋势图.png'), dpi=150)
        plt.close()


if __name__ == '__main__':
    analyzer = MovieAnalyzer()

    print('=' * 50)
    print('🎉 豆瓣电影分析系统启动（情感分析版）')
    print('=' * 50)

    # 批量处理电影
    for movie_id, movie_name in CONFIG['movies'].items():
        analyzer.analyze_movie(movie_id, movie_name)

    print('\n✅ 分析完成！结果已保存至reports目录')