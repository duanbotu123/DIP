from alive_progress import alive_bar # 这个导入通常总是有效的
import time
import random

total_tasks = 100
print("\n运行一个示例 alive_bar...")
with alive_bar(total_tasks,
               title="🚀 最终测试...",
               length=60,
               max_cols=120,
               theme="smooth",
               # 你可以从上面打印出的列表中选择 spinner 和 bar 的名字
               spinner="loving", # 这是一个常见的 spinner
               bar="halloween"    # 这是一个常见的 bar
              ) as bar:
    for i in range(total_tasks):
        time.sleep(0.2)
        bar()
print("最终测试完成!")


# from alive_progress import show_bars
# show_bars()

# from alive_progress.styles import showtime, Show

# showtime(Show.BARS)
# showtime(Show.THEMES)
# showtime(Show.SPINNERS)