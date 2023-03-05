import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np

green = "#56ae57"
red = "#db5856"
purple = "tab:purple"
blue = "#3399e6"

results_cifar_baseline = 78.27
results_imnet_baseline = 76.55
results_ade20_baseline = 34.08
results_voc12_baseline = 77.68



runtime_cifar_baseline = 2.885
runtime_imnet_baseline = 31.88
runtime_ade20_baseline = 381.4
runtime_voc12_baseline = 670.6

runtime_cifar_16 = np.array([1.517,	1.670,	1.656,	1.679,	1.692,	1.644])
runtime_cifar_64 = np.array([1.529,	1.675,	1.757,	1.696,	1.655,	1.761])
runtime_imnet_16 = np.array([5.773,	6.170,	6.449,	6.452,	6.749,	6.797])
runtime_imnet_64 = np.array([6.278,	7.136,	7.393,	8.144,	8.799,	9.437])
runtime_ade20_32 = np.array([35.39,	39.87,	43.42,	46.83,	52.11,	56.39])
runtime_ade20_64 = np.array([38.29,	46.76,	56.30,	65.11,	73.31,	82.07])
runtime_voc12_32 = np.array([91.25,	101.1,	108.0,	118.4,	129.5,	134.2])
runtime_voc12_64 = np.array([98.51,	116.4,	137.7,	156.8,	181.6,	202.3])

perf_imnet = 100 * np.array([65.59, 70.36, 72.23, 73.30, 74.03, 74.40]) / results_imnet_baseline
perf_ade20 = 100 * np.array([33.23, 34.73, 35.53, 35.77, 36.01, 36.31]) / results_ade20_baseline
perf_voc12 = 100 * np.array([72.23, 74.27, 75.89, 76.26, 76.86, 77.48]) / results_voc12_baseline
perf_cifar = 100 * np.array([65.63, 70.90, 74.62, 74.53, 76.18, 76.86]) / results_cifar_baseline

runtime_boost_imnet = runtime_imnet_baseline / runtime_imnet_16
runtime_boost_cifar = runtime_cifar_baseline / runtime_cifar_16
runtime_boost_ade20 = runtime_ade20_baseline / runtime_ade20_32
runtime_boost_voc12 = runtime_voc12_baseline / runtime_voc12_32

bandwidth_cifar_baseline = 1357235088 * (4/11) / 1000000000
bandwidth_imnet_baseline = 17478334016 * (4/11) / 1000000000
bandwidth_ade20_baseline = 64005736576 / 1000000000
bandwidth_voc12_baseline = 106205953664 / 1000000000
bandwidth_cifar_16 = np.array([566001920, 578446528, 589733760, 601267480, 612290712, 623566328]) * (4/11) / 1000000000
bandwidth_cifar_64 = np.array([579236240, 604916752, 629439888, 654208720, 678467856, 702979376]) * (4/11)  / 1000000000
bandwidth_imnet_16 = np.array([2341120540, 2544648634, 2741103970, 2934551686, 3126430912, 3318121840]) * (4/11)  / 1000000000
bandwidth_imnet_64 = np.array([2569422460, 3001253266, 3426011314, 3847761742, 4267943680, 4687915408]) * (4/11)  / 1000000000
bandwidth_ade20_32 = np.array([6341935552, 7542829312, 8724150656, 9879281600, 11035320000, 12158112960]) / 1000000000
bandwidth_ade20_64 = np.array([7004879488, 8858244736, 10692626048, 12500557952, 14310337152, 16085475456]) / 1000000000
bandwidth_voc12_32 = np.array([12305108608, 14874560576, 17276690368, 19699197824, 22078597056, 24376760064]) / 1000000000
bandwidth_voc12_64 = np.array([13765095040, 17783405696, 21574989952, 25407395456, 29186553984, 32841516672]) / 1000000000


bandwidth_boost_imnet = bandwidth_imnet_baseline / bandwidth_imnet_16
bandwidth_boost_cifar = bandwidth_cifar_baseline / bandwidth_cifar_16
bandwidth_boost_ade20 = bandwidth_ade20_baseline / bandwidth_ade20_32
bandwidth_boost_voc12 = bandwidth_voc12_baseline / bandwidth_voc12_32

ticks_font_size = 16
axis_label_font_size = 20
plot_lw = 5
plot_markersize = 15

plt.figure(figsize=(8, 12))
plt.subplot(211)
plt.plot(perf_ade20, bandwidth_boost_ade20, ".-", color=green, lw=plot_lw, markersize=plot_markersize, label="ADE20K - Seg.")
plt.plot(perf_voc12, bandwidth_boost_voc12, ".-", color=purple, lw=plot_lw, markersize=plot_markersize, label="VOC12 - Seg.")
plt.plot(perf_imnet, bandwidth_boost_imnet, ".-", color=blue, lw=plot_lw, markersize=plot_markersize, label="ImageNet - Cls.")
plt.plot(perf_cifar, bandwidth_boost_cifar, ".-", color=red, lw=plot_lw, markersize=plot_markersize, label="CIFAR100 - Cls.")
plt.plot(perf_ade20[1:2], bandwidth_boost_ade20[1:2], ".-", color=green, lw=plot_lw, markersize=25)
plt.plot(perf_voc12[1:2], bandwidth_boost_voc12[1:2], ".-", color=purple, lw=plot_lw, markersize=25)
plt.plot(perf_imnet[1:2], bandwidth_boost_imnet[1:2], ".-", color=blue, lw=plot_lw, markersize=25)
plt.plot(perf_cifar[1:2], bandwidth_boost_cifar[1:2], ".-", color=red, lw=plot_lw, markersize=25)

peft = [perf_ade20[1], perf_voc12[1], perf_imnet[1], perf_cifar[1]]
bandwidth_boost = [bandwidth_boost_ade20[1], bandwidth_boost_voc12[1], bandwidth_boost_imnet[1], bandwidth_boost_cifar[1]]
bandwidth = [bandwidth_ade20_32[1], bandwidth_voc12_32[1], bandwidth_imnet_16[1], bandwidth_cifar_16[1]]
bandwidth = [str(round(x, 2)) for x in bandwidth]
ax = plt.gca()
for i in range(4):
    ax.annotate(str(bandwidth[i]) + "GB", (peft[i]+0.25, bandwidth_boost[i]+0.25), fontsize=14, weight='bold')



ax = plt.gca()

ax.set_ylabel('Factor in Bandwidth', fontsize=axis_label_font_size, labelpad=7)
[i.set_linewidth(2) for i in ax.spines.values()]
ax.set_ylim([1, 11.0])
ax.set_xlim([90, 108])
ax.set_yticklabels([None, "x2", "x4", "x6", "x8", "x10"], fontsize=ticks_font_size)
ax.set_xticklabels([None] + [str(x) + "%" for x in np.arange(92, 107, 2)], fontsize=ticks_font_size)
ax.set_xlabel('Performance Relative to Baseline', fontsize=axis_label_font_size, labelpad=22)
plt.minorticks_on()
ax.grid(visible=True, which='major', color='#666666', linestyle='--', alpha=1, lw=1, axis='y')
ax.grid(visible=True, which='minor', color='#999999', linestyle='--', alpha=0.5, lw=1, axis='y')
ax.grid(visible=True, which='minor', color='#999999', linestyle='--', alpha=0.5, lw=1, axis='x')
ax.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.8, lw=2, axis='x')
plt.legend(prop={'size': 16}, loc='lower right')


plt.subplot(212)

plt.plot(perf_ade20, runtime_boost_ade20, ".-", color=green, lw=plot_lw, markersize=plot_markersize, label="ADE20K - Seg.")
plt.plot(perf_voc12, runtime_boost_voc12, ".-", color=purple, lw=plot_lw, markersize=plot_markersize, label="VOC12 - Seg.")
plt.plot(perf_imnet, runtime_boost_imnet, ".-", color=blue, lw=plot_lw, markersize=plot_markersize, label="ImageNet - Cls.")
plt.plot(perf_cifar, runtime_boost_cifar, ".-", color=red, lw=plot_lw, markersize=plot_markersize, label="CIFAR100 - Cls.")

plt.plot(perf_ade20[1:2], runtime_boost_ade20[1:2], ".-", color=green, lw=plot_lw, markersize=25)
plt.plot(perf_voc12[1:2], runtime_boost_voc12[1:2], ".-", color=purple, lw=plot_lw, markersize=25)
plt.plot(perf_imnet[1:2], runtime_boost_imnet[1:2], ".-", color=blue, lw=plot_lw, markersize=25)
plt.plot(perf_cifar[1:2], runtime_boost_cifar[1:2], ".-", color=red, lw=plot_lw, markersize=25)

peft = [perf_ade20[1], perf_voc12[1], perf_imnet[1], perf_cifar[1]]

runtime_boost = [runtime_boost_ade20[1], runtime_boost_voc12[1], runtime_boost_imnet[1], runtime_boost_cifar[1]]
runtime = [runtime_ade20_32[1], runtime_voc12_32[1], runtime_imnet_16[1], runtime_cifar_16[1]]
runtime = [str(round(x, 2)) for x in runtime]
ax = plt.gca()
for i in range(4):
    ax.annotate(str(runtime[i]) + "s", (peft[i]+0.25, runtime_boost[i]+0.25), fontsize=14, weight='bold')


ax = plt.gca()

# for i in range(len(classification_runtime)):
#     if i != 2:
#         ax.annotate(str(classification_runtime[i]) + "s", (classification_results_24_bits[i]+0.25, classification_runtime_lan_24_bits[i]+0.25), fontsize=14, weight='bold')
#
#
# for i in range(len(segmentation_runtime)):
#     if i != 0:
#         ax.annotate(str(segmentation_runtime[i]) + "s", (segmentation_results_24_bits[i]+0.25, segmentation_runtime_lan_24_bits[i]+0.25), fontsize=14, weight='bold')
#

ax.set_ylabel('Factor in Runtime', fontsize=axis_label_font_size, labelpad=7)
[i.set_linewidth(2) for i in ax.spines.values()]
ax.set_ylim([1, 11.0])
ax.set_xlim([90, 108])
ax.set_yticklabels([None, "x2", "x4", "x6", "x8", "x10"], fontsize=ticks_font_size)
ax.set_xticklabels([None] + [str(x) + "%" for x in np.arange(92, 107, 2)], fontsize=ticks_font_size)
ax.set_xlabel('Performance Relative to Baseline', fontsize=axis_label_font_size, labelpad=22)
plt.minorticks_on()
ax.grid(visible=True, which='major', color='#666666', linestyle='--', alpha=1, lw=1, axis='y')
ax.grid(visible=True, which='minor', color='#999999', linestyle='--', alpha=0.5, lw=1, axis='y')
ax.grid(visible=True, which='minor', color='#999999', linestyle='--', alpha=0.5, lw=1, axis='x')
ax.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.8, lw=2, axis='x')
plt.legend(prop={'size': 16}, loc='lower right')


plt.subplots_adjust(left=0.12, right=0.96, top=0.97, bottom=0.1, hspace=0.15)

plt.savefig("/home/yakir/Figure_bandwidth_and_runtime.png")
