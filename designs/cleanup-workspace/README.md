# 清理工作区交互原型

2026-09-12 已由用户确认，用于第一次交付的正式页面接入。
入口为 index.html，场景包括普通字幕、片头无字幕、多个目标、试擦失败与重试。

所有处理状态为模拟。原片来自 input/detext_examples/english1.mp4 和 others.mp4；
清理样片来自既有 result/speed-auto-full/<sample>/run-001/<sample>_clean.mp4。
上下对比是这两条视频的同步堆叠，不随原型中的选择或区域框改变。
片头无字幕是构造案例：前 12 秒使用既有清理结果，后续使用原片。
目标裁剪图来自每个本地示例的对应帧；原型中的时间区间是演示数据。

assets 下的媒体和 sources.json 是本机审阅素材，按仓库约定不进入 git。
在仓库根目录使用支持 Range 的本地服务打开，以便视频能正确寻址：

```sh
python -c 'from starlette.staticfiles import StaticFiles; import uvicorn; uvicorn.run(StaticFiles(directory="designs"),host="127.0.0.1",port=4311)'
```

访问 http://127.0.0.1:4311/cleanup-workspace/index.html。
正式页面在 src/videowipe/web/index.html，通过 videowipe serve 使用真实任务。

浏览器检查覆盖：1280×800 与 390×844、明暗配色、键盘启动和区域调整、
四条场景、空选择禁用、保留目标定位、改框后的试擦失效，以及失败重试。
