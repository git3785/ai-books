
// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Getting Started',
      items: ['index'],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 1: Basics',
      items: [
        'Module-1-Basics/Overview',
        'Module-1-Basics/Learning-Outcomes',
        'Module-1-Basics/References',
        {
          type: 'category',
          label: 'Tutorials',
          key: 'module-1-tutorials',
          items: [
            'Module-1-Basics/Tutorials/introduction',
          ],
        },
        {
          type: 'category',
          label: 'Diagrams',
          key: 'module-1-diagrams',
          items: [
            'Module-1-Basics/Diagrams/basic-architecture',
          ],
        },
        {
          type: 'category',
          label: 'Code Examples',
          key: 'module-1-code-examples',
          items: [
            'Module-1-Basics/Code-Examples/basic-example',
          ],
        },
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 2: Intermediate',
      items: [
        'Module-2-Intermediate/Overview',
        'Module-2-Intermediate/Learning-Outcomes',
        'Module-2-Intermediate/References',
        {
          type: 'category',
          label: 'Tutorials',
          key: 'module-2-tutorials',
          items: [
            'Module-2-Intermediate/Tutorials/advanced-topics',
          ],
        },
        {
          type: 'category',
          label: 'Diagrams',
          key: 'module-2-diagrams',
          items: [
            'Module-2-Intermediate/Diagrams/system-design',
          ],
        },
        {
          type: 'category',
          label: 'Code Examples',
          key: 'module-2-code-examples',
          items: [
            'Module-2-Intermediate/Code-Examples/intermediate-example',
          ],
        },
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 3: Advanced',
      items: [
        'Module-3-Advanced/Overview',
        'Module-3-Advanced/Learning-Outcomes',
        'Module-3-Advanced/References',
        {
          type: 'category',
          label: 'Tutorials',
          key: 'module-3-tutorials',
          items: [
            'Module-3-Advanced/Tutorials/masterclass',
          ],
        },
        {
          type: 'category',
          label: 'Diagrams',
          key: 'module-3-diagrams',
          items: [
            'Module-3-Advanced/Diagrams/advanced-flow',
          ],
        },
        {
          type: 'category',
          label: 'Code Examples',
          key: 'module-3-code-examples',
          items: [
            'Module-3-Advanced/Code-Examples/advanced-example',
          ],
        },
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 4: VLA Integration',
      items: [
        'Module-4-VLA/Overview',
        'Module-4-VLA/Learning-Outcomes',
        'Module-4-VLA/References',
        {
          type: 'category',
          label: 'Tutorials',
          key: 'module-4-tutorials',
          items: [
            'Module-4-VLA/Tutorials/edge-deployment',
            'Module-4-VLA/Tutorials/whisper-integration',
            'Module-4-VLA/Tutorials/nlp-pipeline',
            'Module-4-VLA/Tutorials/ros2-action-mapping',
            'Module-4-VLA/Tutorials/multi-modal-perception',
          ],
        },
        {
          type: 'category',
          label: 'Diagrams',
          key: 'module-4-diagrams',
          items: [
            'Module-4-VLA/Diagrams/system-architecture',
            'Module-4-VLA/Diagrams/vla-workflow',
          ],
        },
        {
          type: 'category',
          label: 'Code Examples',
          key: 'module-4-code-examples',
          items: [
            'Module-4-VLA/Code-Examples/example-1',
            'Module-4-VLA/Code-Examples/example-2',
          ],
        },
      ],
      collapsed: false,
    },
  ],
};

module.exports = sidebars;
