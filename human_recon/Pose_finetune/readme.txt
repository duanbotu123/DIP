smpl_full: 估出来的待优化pose序列
interaction: 待优化人和物体mesh
codes: 之前的部分代码，仅作参考

TODO：
按照计划，加入以下5项Loss,并整合代码


1. VPoser (human body prior)
施加于每一帧人体动作，纠正不自然的人体动作。
https://github.com/nghorbani/human_body_prior?tab=readme-ov-file#tutorials
2. Contact
施加于每一帧人、物动作，尽量使得人和物体表面贴合。
https://github.com/YinghaoHuang91/InterCap/tree/master
https://github.com/YinghaoHuang91/InterCap/tree/master/Data/body_segments
https://github.com/YinghaoHuang91/InterCap/blob/master/prox_first/prox/fitting.py
3. SDF
施加于每一帧人、物动作，尽量使得人和物体不穿模。和4作用类似。
loss：接触部位SDF不为负
4. Penetration
施加于每一帧人、物动作，尽量使得人和物体不穿模。和3作用类似。
5. Velocity
施加于所有帧人、物动作，尽量使得帧与帧之间动作连续。


