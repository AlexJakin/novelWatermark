# task

1. 不同方向的水印
2. 不同水印攻击
3. 不同llm模型
4. 不同文本长度
4. 不同语言


# 实验一

gpt3 2000样本 gen_len = 30
Human-generated text correctly classified: 100.00%
Watermarked text correctly classified: 94.84%
Non-watermarked text correctly classified: 80.98%


Human-generated text correctly classified: 100.00%
Watermarked text correctly classified: 95.14%
Non-watermarked text correctly classified: 80.90%


# 实验二

2024-05-20 01:02:15,645 mylog INFO experiment2.py--70line :The result of the ai direction: Human-generated text correctly classified: 100.00%
2024-05-20 01:20:37,023 mylog INFO experiment2.py--70line :The result of the physical direction: Human-generated text correctly classified: 98.52%
2024-05-20 02:08:15,044 mylog INFO experiment2.py--70line :The result of the economics direction: Human-generated text correctly classified: 99.89%
2024-05-20 02:46:53,429 mylog INFO experiment2.py--70line :The result of the quantum direction: Human-generated text correctly classified: 100.00%
2024-05-20 03:03:47,334 mylog INFO experiment2.py--70line :The result of the math direction: Human-generated text correctly classified: 98.93%

gpt3  gen_len = 30
2024-05-20 19:23:42,082 mylog INFO experiment2.py--70line :The result of the ai direction: Human-generated text correctly classified: 100.00%
2024-05-20 19:50:04,449 mylog INFO experiment2.py--70line :The result of the physical direction: Human-generated text correctly classified: 98.00%
2024-05-20 21:22:19,400 mylog INFO experiment2.py--70line :The result of the economics direction: Human-generated text correctly classified: 99.89%
2024-05-20 22:48:01,393 mylog INFO experiment2.py--70line :The result of the quantum direction: Human-generated text correctly classified: 100.00%
2024-05-20 23:26:45,947 mylog INFO experiment2.py--70line :The result of the math direction: Human-generated text correctly classified: 99.62%

gpt3  gen_len = 30
2024-05-21 02:22:18,016 mylog INFO experiment4.py--68line :The result of the ai direction: Watermarked text correctly classified: 89.57%
2024-05-21 03:16:52,637 mylog INFO experiment4.py--68line :The result of the physical direction: Watermarked text correctly classified: 88.73%
2024-05-21 05:39:59,077 mylog INFO experiment4.py--68line :The result of the economics direction: Watermarked text correctly classified: 90.07%
2024-05-21 07:52:51,494 mylog INFO experiment4.py--68line :The result of the quantum direction: Watermarked text correctly classified: 89.43%
2024-05-21 09:15:03,134 mylog INFO experiment4.py--68line :The result of the math direction: Watermarked text correctly classified: 85.26%

gpt3  gen_len = 100
2024-05-21 20:40:35,838 mylog INFO experiment4.py--68line :The result of the ai direction: Watermarked text correctly classified: 99.90%
2024-05-22 00:34:18,826 mylog INFO experiment4.py--68line :The result of the physical direction: Watermarked text correctly classified: 100.00%
2024-05-22 10:47:00,883 mylog INFO experiment4.py--68line :The result of the economics direction: Watermarked text correctly classified: 99.95%
2024-05-22 20:19:22,747 mylog INFO experiment4.py--68line :The result of the quantum direction: Watermarked text correctly classified: 99.88%
2024-05-23 02:08:52,746 mylog INFO experiment4.py--68line :The result of the math direction: Watermarked text correctly classified: 100.00%

