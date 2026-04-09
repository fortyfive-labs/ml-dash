import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'ML-Dash',
  tagline: 'Simple and flexible SDK for ML experiment tracking and data storage',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://docs.dash.ml',
  baseUrl: '/',

  headTags: [
    {
      tagName: 'meta',
      attributes: {
        name: 'algolia-site-verification',
        content: '80B062FABBF268CA',
      },
    },
  ],

  organizationName: 'fortyfive-labs',
  projectName: 'ml-dash',

  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/fortyfive-labs/ml-dash/tree/main/website/',
          routeBasePath: '/docs',
          lastVersion: 'current',
          versions: {
            current: {
              label: '0.6.25',
            },
          },
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'ML-Dash',
      logo: {
        alt: 'ML-Dash Logo',
        src: '/img/logo.png',
        srcDark: '/img/logo.png',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar',
          position: 'left',
          label: 'Docs',
        },
        {
          type: 'docSidebar',
          sidebarId: 'examplesSidebar',
          position: 'left',
          label: 'Examples',
        },
        {
          type: 'docSidebar',
          sidebarId: 'apiSidebar',
          position: 'left',
          label: 'Reference',
        },
        {
          type: 'docsVersionDropdown',
          position: 'right',
          dropdownActiveClassDisabled: true,
        },
        {
          href: 'https://pypi.org/project/ml-dash/',
          label: 'PyPI',
          position: 'right',
        },
        {
          href: 'https://github.com/fortyfive-labs/ml-dash',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['bash', 'python'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
