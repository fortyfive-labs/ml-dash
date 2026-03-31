import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docsSidebar: [
    {type: 'doc', id: 'index', label: 'Introduction'},
    {type: 'doc', id: 'getting-started', label: 'Getting Started'},
    {
      type: 'category',
      label: 'Core',
      items: ['experiments', 'parameters', 'metrics', 'logging', 'files'],
    },
    {
      type: 'category',
      label: 'Advanced',
      items: ['buffering', 'tracks', 'images'],
    },
  ],
  apiSidebar: [
    'api-reference',
    {type: 'doc', id: 'cli', label: 'CLI'},
  ],
  examplesSidebar: [
    'examples/simple-training',
    'examples/pytorch-mnist',
    'examples/hyperparameter-search',
    'examples/experiment-comparison',
    'examples/logging-debugging',
  ],
};

export default sidebars;
