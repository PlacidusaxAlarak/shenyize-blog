# AGENTS.md

## 项目是什么
- 这是 `shenyize.com` 的主站仓库，技术栈是 `Astro 5 + Swup + Tailwind + Svelte islands`。
- 博客文章详情页在 `src/pages/posts/[...slug].astro`。
- 站点已经有一个独立的滑动验证码 demo，静态托管在 `public/slider-captcha/`，项目页会链接到它。
- 当前这轮工作的目标不是改那个独立 demo，而是把它的核心验证码逻辑复用到 `posts` 文章详情页里，做成“页内弹层验证码”。

## 当前任务的准确目标
- 用户第一次打开任意 `posts` 文章时，直接在当前页面上方弹出一个滑动验证码层。
- 不跳转到新网址，不进入单独的 challenge 页面。
- 验证通过后，本标签页内继续访问其他 `posts` 文章时不再重复弹出。
- 这是前端交互门禁，不是服务端正文保护。
- `solutions`、首页、归档页、关于页、项目页都不在本次作用范围内。

## 已经做到哪里了

### 已完成的实现
- 已新增文章门禁组件：
  - `src/components/article-captcha/ArticleCaptchaGate.astro`
- 已新增验证码运行时脚本：
  - `src/scripts/article-captcha/index.ts`
  - `src/scripts/article-captcha/logic.mjs`
  - `src/scripts/article-captcha/interactions.mjs`
  - `src/scripts/article-captcha/renderer.mjs`
  - 以及对应的 `.d.ts` 文件
- 已新增站点内固定背景资源：
  - `public/captcha/demo-background.svg`
  - `public/captcha/placeholder-background.svg`
- 已在 `posts` 详情页模板里接入门禁组件：
  - `src/pages/posts/[...slug].astro`
- 已在全局布局里接入 bootstrap：
  - `src/layouts/Layout.astro`
- 已新增测试：
  - `test/article-captcha-gate.test.mjs`

### 当前实现的行为
- 门禁组件通过 `data-*` 属性暴露运行时配置。
- 默认 `sessionStorage` key 是 `article-captcha:posts`。
- 当前标签页通过一次后，后续 `posts` 页面直接放行。
- 刷新验证码会重出题，保留：
  - 五边形拼块
  - 真实槽位
  - 迷惑槽位
  - 随机旋转
  - 容差校验
  - 失败回弹
- `Swup` 页面切换后会重新扫描门禁节点，避免只在首次加载时生效。

## 涉及到的文件

### 新增文件
- `src/components/article-captcha/ArticleCaptchaGate.astro`
- `src/scripts/article-captcha/index.ts`
- `src/scripts/article-captcha/logic.mjs`
- `src/scripts/article-captcha/logic.d.ts`
- `src/scripts/article-captcha/interactions.mjs`
- `src/scripts/article-captcha/interactions.d.ts`
- `src/scripts/article-captcha/renderer.mjs`
- `src/scripts/article-captcha/renderer.d.ts`
- `public/captcha/demo-background.svg`
- `public/captcha/placeholder-background.svg`
- `test/article-captcha-gate.test.mjs`

### 修改文件
- `src/pages/posts/[...slug].astro`
- `src/layouts/Layout.astro`

## 关键实现说明

### `ArticleCaptchaGate.astro`
- 负责输出门禁 DOM 结构。
- 内含：
  - 遮罩层
  - 卡片容器
  - Canvas
  - range slider
  - refresh button
  - status / meta 文案
- 文章正文通过 slot 包裹在组件内。
- 默认配置：
  - `storageKey = "article-captcha:posts"`
  - `backgroundImageUrl = "/captcha/demo-background.svg"`
  - `fallbackBackgroundImageUrl = "/captcha/placeholder-background.svg"`

### `src/scripts/article-captcha/index.ts`
- 负责扫描 `[data-article-captcha-gate]` 并挂载控制器。
- 管理门禁的锁定/解锁状态。
- 管理 `sessionStorage` 放行状态。
- 处理 `Swup` 的 `page:view` 生命周期。
- 内部通过 `controllers` Map 避免重复初始化。

### `logic.mjs`
- 直接复用了独立 demo 的几何逻辑思路。
- 目前保留的是纯逻辑能力，方便 `node:test` 直接 import 验证。

### `interactions.mjs`
- 负责 slider 的输入、释放、键盘和 pointer 事件。
- 这里和原 demo 的区别是增加了 `destroy()`，用于 `Swup` 场景下清理文档级事件，避免页面切换后重复绑定。

### `renderer.mjs`
- 负责：
  - 背景图加载
  - notch / piece 绘制
  - Canvas 渲染
- 仍然保持“真实缺口 + 迷惑缺口 + 旋转五边形”的视觉模型。

## 验证状态

### 已验证通过
- `node --test`
  - 通过
  - 当前是 6 个 test 全部通过
- `node --test test/article-captcha-gate.test.mjs`
  - 通过
- `pnpm build`
  - 通过
  - 说明新增代码至少能进入 Astro 生产构建

### 未完成或未全绿
- `pnpm type-check`
  - 仍失败
  - 当前主要剩余错误是仓库已有的 `src/utils/content-utils.ts:100`
- `pnpm check`
  - 仍失败
  - 仓库本身有大量既有的 Astro/content 类型问题
  - 本次改动不是把这个仓库带回全绿基线
- 浏览器级验证
  - 没有完整收尾
  - 原因是上一轮在用浏览器工具做手工验证时被中断了
  - 需要下一终端继续补

## 已知问题与注意事项

### 1. 不要删独立 demo
- `public/slider-captcha/` 必须保留。
- 这是已有项目页功能，不是本次门禁的替代品。

### 2. 不要把问题归因错
- `pnpm check` 和 `pnpm type-check` 目前不是干净基线。
- 这个仓库在本次改动之前就存在一批类型问题。
- 后续如果要修这些问题，应该单独作为一个任务，不要和验证码功能混在一起判断。

### 3. 当前门禁是前端门禁
- 用户浏览器里仍然会拿到正文 HTML。
- 只是正常交互路径上会先被遮罩拦住。
- 如果将来要做“未验证前正文不下发”，需要改成服务端门禁或按需拉取正文。

### 4. `Swup` 是必须考虑的
- 不能只靠 `DOMContentLoaded` 初始化。
- 否则首篇文章可能正常，客户端切到第二篇文章时门禁会失效。

### 5. 当前脚本里有 `@ts-ignore`
- 这是为了让 `index.ts` 能 import `.mjs` 运行时文件，同时保留 `node:test` 直接测试这些模块。
- 如果后续要把类型系统整理干净，可以考虑：
  - 把 `.mjs` 迁成 `.ts`
  - 或补统一的 module declaration
  - 或重新设计测试入口

## 下一终端优先做什么

### 推荐优先级
1. 先做浏览器级验证
2. 确认弹层视觉与交互没有问题
3. 再决定是否需要继续整理类型系统或文案

### 建议检查的具体场景
- 首次打开 `posts` 文章是否立即出现验证码弹层
- 未通过前正文是否不可交互
- 验证成功后是否不刷新页面直接解锁
- 同一标签页跳转到另一篇 `posts` 文章时是否不再弹出
- 新标签页重新打开文章时是否重新弹出
- 非 `posts` 页面是否不会误弹
- 刷新验证码按钮是否始终可用
- 失败回弹动画是否正常

## 新终端接手命令

### 进入仓库
```powershell
Set-Location G:\shenyize-blog
```

### 先看当前改动
```powershell
git status --short
```

### 跑测试
```powershell
node --test
```

### 单独跑这次新增的门禁测试
```powershell
node --test test/article-captcha-gate.test.mjs
```

### 本地开发
```powershell
pnpm dev
```

### 生产构建验证
```powershell
pnpm build
```

### 类型检查
```powershell
pnpm type-check
```

### Astro 检查
```powershell
pnpm check
```

## 建议的继续操作顺序

### 如果你要继续开发
```powershell
Set-Location G:\shenyize-blog
git status --short
node --test
pnpm dev
```

然后打开：
- `http://localhost:4321/posts/guild/`
- 再手动切几篇其他文章验证门禁是否按预期工作

### 如果你要做构建产物验证
```powershell
Set-Location G:\shenyize-blog
pnpm build
pnpm preview
```

或者你也可以用静态目录服务 `dist/` 做验证。

## 之前遗留的本地验证状态
- 上一轮为了看构建产物，曾启动过一个本地静态服务，地址是 `http://127.0.0.1:4322/`。
- 如果你发现这个端口仍然能访问，说明之前的本地服务进程还活着。
- 如果你不想用它，可以自己重新起服务，或者先查占用：

```powershell
Get-NetTCPConnection -LocalPort 4322 -ErrorAction SilentlyContinue
```

## 如果你要继续改代码，推荐从哪里看
- 门禁结构和样式：`src/components/article-captcha/ArticleCaptchaGate.astro`
- 挂载逻辑和 session 状态：`src/scripts/article-captcha/index.ts`
- 纯几何逻辑：`src/scripts/article-captcha/logic.mjs`
- 文章模板接入点：`src/pages/posts/[...slug].astro`
- 全局注入点：`src/layouts/Layout.astro`
- 回归测试：`test/article-captcha-gate.test.mjs`

## Repository Guidelines

### Project Structure & Module Organization
- This is an Astro-based blog/site.
- Source code lives in `src/` with `components/`, `layouts/`, `pages/`, `styles/`, `utils/`, `plugins/`, `i18n/`, and `types/`.
- Content collections are in `src/content/` and posts live in `src/content/posts/`.
- Static assets go in `public/`.
- Tooling scripts live in `scripts/`.
- Generated output goes to `dist/`.

### Build, Test, and Development Commands
- `pnpm install`: install deps. Node >=20 and pnpm >=9 are required.
- `pnpm dev` / `pnpm start`: local dev server at `http://localhost:4321`.
- `pnpm build`: production build plus Pagefind index into `dist/`.
- `pnpm preview`: serve the built site.
- `pnpm check`: Astro checks for content/types.
- `pnpm type-check`: run `tsc` without emit.
- `pnpm lint`: Biome checks and auto-fixes `src/`.
- `pnpm format`: Biome format `src/`.

### Coding Style & Naming Conventions
- Formatting is handled by Biome with tabs and double quotes.
- Organize imports via Biome actions.
- Prefer kebab-case filenames for posts.
- Keep frontmatter fields consistent with `frontmatter.json`.

### Testing Guidelines
- Treat `pnpm check`, `pnpm type-check`, and `pnpm lint` as required pre-PR checks.
- For visual changes, verify locally with `pnpm dev` and `pnpm build`.

### Commit & Pull Request Guidelines
- Prefer Conventional Commits when possible.
- Keep PRs focused on a single purpose.
- UI 改动尽量附截图或录屏。

### Configuration & Content Tips
- Primary site settings live in `src/config.ts`.
- Deployment URL and integrations are in `astro.config.mjs`.
- If you modify frontmatter fields, update `frontmatter.json` and `src/content/config.ts` together.
