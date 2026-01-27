// astro.config.mjs
import sitemap from "@astrojs/sitemap";
import svelte from "@astrojs/svelte";
import tailwind from "@astrojs/tailwind";
import { pluginCollapsibleSections } from "@expressive-code/plugin-collapsible-sections";
import { pluginLineNumbers } from "@expressive-code/plugin-line-numbers";
import swup from "@swup/astro";
import expressiveCode from "astro-expressive-code";
import icon from "astro-icon";
import { defineConfig } from "astro/config";
import rehypeAutolinkHeadings from "rehype-autolink-headings";
import rehypeComponents from "rehype-components";
import rehypeKatex from "rehype-katex";
import rehypeSlug from "rehype-slug";
import remarkDirective from "remark-directive";
import remarkGithubAdmonitionsToDirectives from "remark-github-admonitions-to-directives";
import remarkMath from "remark-math";
import remarkSectionize from "remark-sectionize";
import { expressiveCodeConfig } from "./src/config.ts";
import { pluginLanguageBadge } from "./src/plugins/expressive-code/language-badge.ts";
import { AdmonitionComponent } from "./src/plugins/rehype-component-admonition.mjs";
import { GithubCardComponent } from "./src/plugins/rehype-component-github-card.mjs";
import { parseDirectiveNode } from "./src/plugins/remark-directive-rehype.js";
import { remarkExcerpt } from "./src/plugins/remark-excerpt.js";
import { remarkReadingTime } from "./src/plugins/remark-reading-time.mjs";
import { pluginCustomCopyButton } from "./src/plugins/expressive-code/custom-copy-button.js";

/* New: 引入 execSync 用于执行生成脚本 */
import { execSync } from "node:child_process";

/* New: 定义自动生成题解索引的 Vite 插件 */
function autoGenerateSolutions() {
    return {
        name: "auto-generate-solutions",
        // 1. 在服务器启动或构建开始时运行一次
        buildStart() {
            try {
                // 假设你的脚本在 scripts/generate-solutions.js
                console.log("正在生成题解索引...");
                execSync("node scripts/generate-solutions.js", { stdio: "inherit" });
            } catch (e) {
                console.error("题解索引生成失败:", e);
            }
        },
        // 2. 监听文件热更新
        handleHotUpdate({ file }) {
            // 只有当 Atcoder 或 Codeforces 目录下的文件发生变化时才触发
            if (
                (file.includes("/posts/Atcoder/") || file.includes("/posts/Codeforces/")) &&
                file.endsWith(".md")
            ) {
                console.log("检测到题解变动，正在更新索引...");
                try {
                    execSync("node scripts/generate-solutions.js", { stdio: "inherit" });
                } catch (e) {
                    console.error("题解索引更新失败:", e);
                }
            }
        },
    };
}

// https://astro.build/config
export default defineConfig({
    site: "https://shenyize.com/",
    base: "/",
    trailingSlash: "always",
    integrations: [
        tailwind({ nesting: true }),
        swup({
            theme: false,
            animationClass: "transition-swup-",
            containers: ["main", "#toc"],
            smoothScrolling: true,
            cache: true,
            preload: true,
            accessibility: true,
            updateHead: true,
            updateBodyClass: false,
            globalInstance: true,
        }),
        icon({
            include: {
                "preprocess: vitePreprocess(),": ["*"],
                "fa6-brands": ["*"],
                "fa6-regular": ["*"],
                "fa6-solid": ["*"],
            },
        }),
        expressiveCode({
            themes: [expressiveCodeConfig.theme, expressiveCodeConfig.theme],
            plugins: [
                pluginCollapsibleSections(),
                pluginLineNumbers(),
                pluginLanguageBadge(),
                pluginCustomCopyButton(),
            ],
            defaultProps: {
                wrap: true,
                overridesByLang: {
                    shellsession: {
                        showLineNumbers: false,
                    },
                },
            },
            styleOverrides: {
                codeBackground: "var(--codeblock-bg)",
                borderRadius: "0.75rem",
                borderColor: "none",
                codeFontSize: "0.875rem",
                codeFontFamily:
                    "'JetBrains Mono Variable', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace",
                
                /* --- 修改开始：解决代码拥挤问题 --- */
                codeLineHeight: "1.75rem",    // 增加行高 (原 1.5rem)
                codePaddingBlock: "1.25rem",  // 增加代码块上下的内边距
                /* --- 修改结束 --- */

                frames: {
                    editorBackground: "var(--codeblock-bg)",
                    terminalBackground: "var(--codeblock-bg)",
                    terminalTitlebarBackground: "var(--codeblock-topbar-bg)",
                    editorTabBarBackground: "var(--codeblock-topbar-bg)",
                    editorActiveTabBackground: "none",
                    editorActiveTabIndicatorBottomColor: "var(--primary)",
                    editorActiveTabIndicatorTopColor: "none",
                    editorTabBarBorderBottomColor: "var(--codeblock-topbar-bg)",
                    terminalTitlebarBorderBottomColor: "none",
                },
                textMarkers: {
                    delHue: 0,
                    insHue: 180,
                    markHue: 250,
                },
            },
            frames: {
                showCopyToClipboardButton: false,
            },
        }),
        svelte(),
        sitemap(),
    ],
    markdown: {
        remarkPlugins: [
            remarkMath,
            remarkReadingTime,
            remarkExcerpt,
            remarkGithubAdmonitionsToDirectives,
            remarkDirective,
            remarkSectionize,
            parseDirectiveNode,
        ],
        rehypePlugins: [
            rehypeKatex,
            rehypeSlug,
            [
                rehypeComponents,
                {
                    components: {
                        github: GithubCardComponent,
                        note: (x, y) => AdmonitionComponent(x, y, "note"),
                        tip: (x, y) => AdmonitionComponent(x, y, "tip"),
                        important: (x, y) => AdmonitionComponent(x, y, "important"),
                        caution: (x, y) => AdmonitionComponent(x, y, "caution"),
                        warning: (x, y) => AdmonitionComponent(x, y, "warning"),
                    },
                },
            ],
            [
                rehypeAutolinkHeadings,
                {
                    behavior: "append",
                    properties: {
                        className: ["anchor"],
                    },
                    content: {
                        type: "element",
                        tagName: "span",
                        properties: {
                            className: ["anchor-icon"],
                            "data-pagefind-ignore": true,
                        },
                        children: [
                            {
                                type: "text",
                                value: "#",
                            },
                        ],
                    },
                },
            ],
        ],
    },
    vite: {
        /* New: 注册自定义插件 */
        plugins: [autoGenerateSolutions()],
        build: {
            rollupOptions: {
                onwarn(warning, warn) {
                    if (
                        warning.message.includes("is dynamically imported by") &&
                        warning.message.includes("but also statically imported by")
                    ) {
                        return;
                    }
                    warn(warning);
                },
            },
        },
    },
});