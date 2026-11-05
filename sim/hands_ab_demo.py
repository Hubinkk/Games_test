"""
A+B 混合方案（每只手 6 个电机 + 6 个锁）的十指打字测试

用法：
    python hands_ab_demo.py      # 逐档测速度，并把不出错的最快一档导出到 ../hands_ab_frames.js

和 14 电机版的区别全部来自机构：
  - 食指、中指为差动双线，中间节与根部联动，所以每根手指只有"张开"和"弯曲"两个可控量
  - 无名指、小指、张开、拇指对掌、拇指弯曲共用一个电机：
    同一时刻只能驱动其中一个，换到另一个输出要先切换离合器（T_SW）；没被驱动的输出由锁保持
  - 拇指只保留"对掌摆动"和"弯曲"；悬停、按下、抬起都只改弯曲，减少分时切换
  - 调度器会为分时驱动预留额外的移动时间
"""
import hands_ab_build as hab
import hands_demo as hd

T_SW = 0.02  # 离合器切换时间（秒）

# 让 hands_demo 使用 A+B 模型
hd.hsb = hab
hd.JOINTS_OF.update({"index": ["index_abd", "index_mcp"], "middle": ["middle_abd", "middle_mcp"],
                     "thumb": ["thumb_opp", "thumb_mcp"]})
hd.COUPLE.update({"index_mcp": [("index_pip", 1.0), ("index_dip", 0.75)],
                  "middle_mcp": [("middle_pip", 1.0), ("middle_dip", 0.75)]})


class HandsAB(hd.Hands):
    TEN_GROUPS = (("index", ["index_a", "index_b"]), ("middle", ["middle_a", "middle_b"]),
                  ("ring", ["ring_flex"]), ("little", ["little_flex", "spread"]),
                  ("thumb", ["thumb_opp", "thumb_flex"]), ("wrist", ["wrist_yaw", "wrist_pitch"]))

    def __init__(self):
        super().__init__()
        self.mux_names = {i: self.m.actuator(i).name[2:] for i in range(self.m.nu)}
        self.switches = 0

    def hover_joints(self, finger):
        return ["thumb_mcp"] if finger == "thumb" else hd.JOINTS_OF[finger]

    def plan(self, text):
        # 右手拇指按空格的右半边（离中线 7 mm），右手不必往中间挤，避免和左手拇指相撞
        x, _ = self.keys[" "]
        self.keys[" "] = (x, -0.007)
        strokes = super().plan(text)
        last = {}
        for s in strokes:  # 记下同一只手上一次用的是哪根手指
            s["prev_finger"] = last.get(s["hand"])
            last[s["hand"]] = s["finger"]
        return strokes

    def extra_move(self, stroke, t_move):
        extra = 0.0
        # 拇指：对掌和弯曲两个输出要依次驱动，各需一次切换；无名指、小指：按下前要切换一次
        if stroke["finger"] == "thumb":
            extra += 2 * T_SW + 0.9 * t_move
        elif stroke["finger"] in ("ring", "little"):
            extra += T_SW
        # 同一只手上一键是无名指或小指：它要先由共享电机抬起，手臂才能移动
        if stroke.get("prev_finger") in ("ring", "little"):
            extra += T_SW + 0.9 * t_move
        return extra

    def type_text(self, text, t_move, t_press, t_gap, label="十指打字（A+B）"):
        self.mux_free = {h: 0.0 for h in hab.SIDES}
        self.mux_last = {h: None for h in hab.SIDES}
        self.switches = 0
        return super().type_text(text, t_move, t_press, t_gap, label)

    def command(self, hand, q, dur):
        """专用电机照常执行；共享电机的各个输出排队，一次只驱动一个，换输出先切换离合器。"""
        now = self.d.time
        if not hasattr(self, "mux_free"):
            self.mux_free = {h: 0.0 for h in hab.SIDES}
            self.mux_last = {h: None for h in hab.SIDES}
        targets = self.ctrl_of(hand, q)
        queue, dedicated = [], []
        for i, v in targets.items():
            name = self.mux_names[i]
            if name in hab.MUXED:
                if abs(v - self.c1[i]) > 1e-7:
                    lifting = name in ("ring_flex", "little_flex") and v < self.c1[i]
                    queue.append((-2 if lifting else hab.MUXED.index(name), i, v))
                continue
            dedicated.append((i, v))
        # 抬起压在键上的无名指或小指优先级最高；它完成之前，手臂和其他专用电机不动（否则会拖过别的键）
        hold_until = now
        lifts = [(k, i, v) for k, i, v in queue if k == -2]
        if lifts:
            start = max(now, self.mux_free[hand])
            for _, i, v in lifts:
                if self.mux_last[hand] != i:
                    start += T_SW
                    self.switches += 1
                self.c0[i], self.c1[i] = self.c1[i], v
                self.ts[i], self.te[i] = start, start + max(dur, 1e-3)
                self.mux_free[hand] = start = self.te[i]
                self.mux_last[hand] = i
            hold_until = self.mux_free[hand]
            queue = [x for x in queue if x[0] != -2]
        for i, v in dedicated:
            self.c0[i], self.c1[i] = self.d.ctrl[i], v
            self.ts[i], self.te[i] = hold_until, hold_until + max(dur, 1e-3)
        # 拇指两个输出的先后（运动学检查得出）：
        #   伸出（对掌角变小）：先在掌下调好弯曲（掌下改弯曲离键面 7 mm 以上），再转出来；
        #     反过来的话，弯曲还停在收起的值，转到空格上方时离键面只有 0.3 mm，会提前按下
        #   收回（对掌角变大）：先转回掌下（此时拇指已抬起，全程离键面 5 mm 以上），再调弯曲
        opp = next(((i, v) for _, i, v in queue if self.mux_names[i] == "thumb_opp"), None)
        if opp and abs(opp[1]) < abs(self.c1[opp[0]]):
            queue = [((-1 if self.mux_names[i] == "thumb_flex" else k), i, v) for k, i, v in queue]
        for _, i, v in sorted(queue):
            start = max(now, self.mux_free[hand])
            if self.mux_last[hand] != i:
                start += T_SW
                self.switches += 1
            self.c0[i], self.c1[i] = self.c1[i], v
            self.ts[i], self.te[i] = start, start + max(dur, 1e-3)
            self.mux_free[hand] = self.te[i]
            self.mux_last[hand] = i


if __name__ == "__main__":
    hab.build()
    print("A+B 混合方案：每只手 6 个电机（食指 2、中指 2、手腕左右 1、共享 1）+ 6 个锁\n")
    hd.main(hands_cls=HandsAB, out_name="hands_ab_frames.js", baseline="14 电机版为 8.3 字符/秒，单指为 3.3 字符/秒")
