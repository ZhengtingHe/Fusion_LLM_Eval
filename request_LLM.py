import toml
with open('config.toml', 'r', encoding='utf-8') as toml_file:
    config = toml.load(toml_file)
from langchain_openai import ChatOpenAI
chat = ChatOpenAI(
    model=config['model'],
    base_url=config['base_url'],
    api_key=config['api_key'],
    temperature=0,
)
messages = """
Given the evaluation steps, return a JSON with two keys: 1) a `score` key ranging from 0 to 10, with 10 being that it follows the criteria outlined in the steps and 0 being that it does not, and 2) a `reason` key, a reason for the given score, but DO NOT QUOTE THE SCORE in your reason. Please mention specific information from Input, Actual Output, and Expected Output in your reason, but be very concise with it!

Evaluation Steps:
1. 用简体中文陈述给分原因
2. 检查最终结论是否与预期中的结论一致
3. 推导思路相近但结论不同的情况，认为是错误的
4. 如果出现公式和物理量，需要检查代表内容是否一致




Input:
考虑一个典型的托卡马克放电过程，从等离子体击穿、电流爬升、平顶到破裂。简述在这些不同阶段，哪些基本的等离子体物理过程（如电离、辐射、输运、不稳定性）占据主导地位？ 

Actual Output:
### 托卡马克放电过程各阶段的主导等离子体物理过程

#### 1. **击穿阶段**
- **主导过程**：电离和电子加热。
- **详细描述**：在这个阶段，中性气体被高电压电场或微波电离，形成等离子体。电子从电极或中性气体中原子中被加速和电离，导致等离子体密度增加。加热过程主要是电子加热，温度相对较低。

#### 2. **电流爬升阶段**
- **主导过程**：初级电子产生、电离、输运和初步的磁场配置。
- **详细描述**：电流迅速增加，等离子体温度和密度上升，磁场增强。电子的热传导和离子的对流开始显著，初级电子的产生和电离继续进行。磁场的配置和稳定性开始受关注，可能开始出现磁流体力学（MHD）不稳定性。

#### 3. **稳态阶段（平顶阶段）**
- **主导过程**：能量平衡、输运、辐射和核聚变。
- **详细描述**：等离子体达到高密度和高温，核聚变反应显著。能量注入与损失（如辐射和输运）达到平衡。输运过程包括经典和非经典热传导，辐射主要为轫致辐射。此阶段可能出现MHD不稳定性，如tearing模和锯齿模，影响等离子体稳定性和性能。

#### 4. **破裂阶段**
- **主导过程**：磁流体力学不稳定性、能量释放和位形变化。
- **详细描述**：等离子体约束被破坏，电流迅速下降。MHD不稳定性导致磁场重新连接，释放大量能量，可能引起等离子体与装置壁接触，导致高热负荷，损坏材料。

### 总结
托卡马克放电过程的各个阶段由不同的等离子体物理过程主导：击穿阶段的电离和加热，电流爬升阶段的电离和输运，稳态阶段的能量平衡和输运，以及破裂阶段的不稳定性。理解这些过程对于优化装置运行、提高聚变效率和装置寿命至关重要。 

Expected Output:
1. **核心结论** —— 在托卡马克放电的不同阶段，各类等离子体物理过程分别主导着等离子体的演化，包括电离、辐射、输运和不稳定性等。

2. **背景与概念**
   - 托卡马克是利用磁约束实现高温等离子体稳定的装置。
   - 放电过程包括四个阶段：等离子体击穿、电流爬升、平顶和破裂。
   - 各阶段涉及等离子体密度、温度、磁场和电流等量的演化。

3. **原理与机理分析**
   - **等离子体击穿**：首先需要足够的电场使气体中的原子电离。这是通过强电场加速少量自由电子，与中性原子碰撞实现的。欧姆加热和电离辐射平衡此过程。
   - **电流爬升**：主要是通过感应电场驱动电流的快速增加，并进一步提升等离子体温度和密度。过程中输运过程（如热输运和粒子输运）显著影响能量和物质分布。
   - **平顶**：电流及等离子体主要参量基本稳定，主要涉及能量和粒子输运；位于平衡态的等离子体需对抗各类不稳定性，确保长时间稳定的运行。
   - **破裂**：由于扰动积累或控制失效，形成磁扰动、垂直位移事件等不稳定性，最终导致等离子体失控而快速消失。

4. **定量估算或示例** —— 常见托卡马克（如JET, ITER）可能涉及电流百万安培量级，等离子体温度百万摄氏度量级，细粒子输运时间尺度可至毫秒量级。

5. **工程或实验意义** —— 各过程影响托卡马克的设计与控制策略，尤其在电流爬升和平顶阶段需重点控制不稳定性。

6. **比较与拓展** —— 不同设计（如仿星器）对这些过程表现不同。仿星器通过复杂磁场结构减少电流爬升阶段的挑战，但设计更加复杂。

7. **要点总结** —— 托卡马克每一阶段涉及具体的等离子体物理过程，设计与控制需考虑其中主导因素，确保等离子体的稳定运行。 



**
IMPORTANT: Please make sure to only return in JSON format, with the "score" and "reason" key. No words or explanation is needed.

Example JSON:
{
    "score": 0,
    "reason": "The text does not follow the evaluation steps provided."
}
**

JSON:
"""
print(chat.invoke(messages))