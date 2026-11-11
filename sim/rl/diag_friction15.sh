# 摩擦 1.5 为什么倒：2026-11-11 做过的全部诊断试验（结果汇总在 diag_friction15.json，结论见 wiki/灵活度.md）
# 用法：bash diag_friction15.sh      （完整身体每次约 2–3 分钟，CPU，十几个一起跑；输出到 diag_out/）
# 注意：简化身体的试验读的是当前的 huaran_rl.xml。第四轮以后它已经是 elliptic + 手指板不碰地；
#       要重现"改之前"的结果，先把 build_rl_model.py 里的 cone="elliptic" 和手指板的 contype/conaffinity 去掉再生成。
cd ~/Games_test/sim/rl
mkdir -p diag_out
P=~/venv-sys/bin/python
POL=run3_final.npz
# 1. 完整身体：1.2（成功）、1.5、1.8（失败）逐 0.1 秒对比
for f in 1.0 1.2 1.5 1.8; do
  $P diag_getup.py full $POL --fric=$f --from=0 --to=6 > diag_out/full_$f.log 2>&1 &
done
# 2. 完整身体 1.5 / 1.8：只把某个部位的摩擦降到 1.0，看是谁"粘住"了
for p in head fing thumb palm fing,thumb,palm thigh; do
  $P diag_getup.py full $POL --fric=1.5 --part=$p:1.0 --from=1.8 --to=3.6 > diag_out/full_1.5_part_$p.log 2>&1 &
done
$P diag_getup.py full $POL --fric=1.8 --part=head:1.0 --from=1.8 --to=4 > diag_out/full_1.8_part_head.log 2>&1 &
# 3. 完整身体换成 pyramidal 摩擦锥（训练用的简化身体第三轮以前用的就是它）
for f in 1.5 1.8; do
  $P diag_getup.py full $POL --fric=$f --cone=pyramidal --from=1.8 --to=4 > diag_out/full_${f}_pyramidal.log 2>&1 &
done
wait
# 4. 简化身体（CPU）：pyramidal / elliptic、手指板碰不碰地
for f in 1.0 1.2 1.5; do
  for c in pyramidal elliptic; do
    $P diag_getup.py simple $POL --fric=$f --cone=$c --k=0.15 --rest=0.15 --from=1.8 --to=3.6 > diag_out/simple_${f}_$c.log 2>&1 &
  done
done
wait
# 5. 可行性：在 1.5–2.7 秒给一组关节加满策略能加的修正（±0.3 弧度），看能不能过"头顶着地"那一步
for g in neck_pitch+neck_low waist_pitch l_elbow+r_elbow l_shoulder_pitch+r_shoulder_pitch l_hip_pitch+r_hip_pitch; do
  for v in 1 -1; do
    $P diag_getup.py full $POL --fric=1.5 --from=2.5 --to=2.5 --bias=$g:$v:1.5:2.7 > diag_out/full_1.5_bias_${g}_$v.log 2>&1 &
  done
done
wait
for x in diag_out/*.log; do echo "$x: $(tail -1 $x)"; done
