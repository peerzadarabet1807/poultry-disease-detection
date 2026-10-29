import classNames from '../data/class_names.json';

export type ClassId = 'Coccidiosis' | 'Healthy' | 'Salmonella' | 'NCD';

/** Model output order (models/class_names.json). */
export const CLASS_IDS = classNames as ClassId[];

export interface ClassInfo {
  id: ClassId;
  name: string;
  short: string;
  /** what the class means, in one plain sentence */
  description: string;
  /** chart colour token (Dark2 in the light theme, Set2 in the dark theme: the notebook's palette family) */
  color: string;
}

export const CLASS_INFO: Record<ClassId, ClassInfo> = {
  Coccidiosis: {
    id: 'Coccidiosis',
    name: 'Coccidiosis',
    short: 'Cocci',
    description: 'A parasitic intestinal disease caused by Eimeria protozoa.',
    color: 'var(--c-cocci)',
  },
  Healthy: {
    id: 'Healthy',
    name: 'Healthy',
    short: 'Healthy',
    description: 'Droppings from birds the dataset labels as healthy.',
    color: 'var(--c-healthy)',
  },
  Salmonella: {
    id: 'Salmonella',
    name: 'Salmonella',
    short: 'Salmo',
    description: 'Infection with Salmonella bacteria (salmonellosis).',
    color: 'var(--c-salmo)',
  },
  NCD: {
    id: 'NCD',
    name: 'Newcastle disease',
    short: 'NCD',
    description: 'A viral disease of birds, notifiable to the veterinary authority in many countries.',
    color: 'var(--c-ncd)',
  },
};

if (CLASS_IDS.length !== 4 || CLASS_IDS.some((id) => !(id in CLASS_INFO))) {
  throw new Error(`Unexpected class names: ${CLASS_IDS.join(', ')}`);
}
