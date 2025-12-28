import {themes as prismThemes} from 'prism-react-renderer';

export default {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'AI-Driven Book on Advanced Robotics Systems',
  url: 'https://your-docusaurus-site.example.com',
  baseUrl: '/',
  onBrokenLinks: 'throw',
  markdown: {
    mermaid: true,
    mdx1Compat: {
      comments: true,
      admonitions: true,
      headingIds: true,
    },
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },
  favicon: 'img/favicon.ico',
  organizationName: 'ai-book',
  projectName: 'physical-ai-humanoid-robotics',
  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'ur'],
    localeConfigs: {
      en: {
        label: 'English',
        direction: 'ltr',
      },
      ur: {
        label: 'اردو',
        direction: 'rtl',
      }
    }
  },
  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          routeBasePath: '/docs',
          showLastUpdateTime: true,
          editUrl: 'https://github.com/ai-book/physical-ai-humanoid-robotics/edit/main/',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],
  themeConfig: {
    image: 'img/book-social-card.jpg', // Social card image
    metadata: [
      {name: 'keywords', content: 'robotics, AI, ROS, NVIDIA Isaac, Gazebo, Unity, Physical AI, Humanoid Robotics, Vision-Language-Action'}
    ],
    colorMode: {
      defaultMode: 'dark',
      disableSwitch: false,
      respectPrefersColorScheme: false,
    },
    navbar: {
      title: 'Physical AI & Humanoid Robotics',
      logo: {
        alt: 'Physical AI & Humanoid Robotics Logo',
        src: 'img/logo.png', // Using existing logo as icon-only
        srcDark: 'img/logo.png', // Using same image for both modes
      },
      items: [
        {
          type: 'dropdown',
          label: 'Modules',
          position: 'left',
          items: [
            {
              type: 'doc',
              docId: 'Module-1-Basics/Overview',
              label: 'Module 1: The Robotic Nervous System (ROS 2)',
            },
            {
              type: 'doc',
              docId: 'Module-2-Intermediate/Overview',
              label: 'Module 2: The Digital Twin (Gazebo & Unity)',
            },
            {
              type: 'doc',
              docId: 'Module-3-Advanced/Overview',
              label: 'Module 3: The AI-Robot Brain (NVIDIA Isaac™)',
            },
            {
              type: 'doc',
              docId: 'Module-4-VLA/Overview',
              label: 'Module 4: Vision-Language-Action (VLA) Integration',
            },
          ],
        },
        {
          type: 'localeDropdown',
          position: 'right',
        },
        {
          href: 'https://github.com/ai-book/physical-ai-humanoid-robotics',
          position: 'right',
          className: 'header-github-link',
          'aria-label': 'GitHub repository',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Modules',
          items: [
            {
              label: 'Module 1: The Robotic Nervous System (ROS 2)',
              to: '/docs/Module-1-Basics/Overview',
            },
            {
              label: 'Module 2: The Digital Twin (Gazebo & Unity)',
              to: '/docs/Module-2-Intermediate/Overview',
            },
            {
              label: 'Module 3: The AI-Robot Brain (NVIDIA Isaac™)',
              to: '/docs/Module-3-Advanced/Overview',
            },
            {
              label: 'Module 4: Vision-Language-Action (VLA) Integration',
              to: '/docs/Module-4-VLA/Overview',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'Stack Overflow',
              href: 'https://stackoverflow.com/questions/tagged/robotics',
            },
            {
              label: 'Robotics Stack Exchange',
              href: 'https://robotics.stackexchange.com/',
            },
            {
              label: 'ROS Discourse',
              href: 'https://discourse.ros.org/',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/ai-book/physical-ai-humanoid-robotics',
            },
            {
              label: 'NVIDIA Developer',
              href: 'https://developer.nvidia.com/robotics',
            },
            {
              label: 'Open Robotics',
              href: 'https://www.openrobotics.org/',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Book. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['bash', 'python', 'cpp', 'json', 'yaml', 'docker', 'csharp'],
    },
  },
};
