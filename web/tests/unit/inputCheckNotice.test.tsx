import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { InputCheckNotice } from '../../src/components/InputCheckNotice';
import type { RejectedPhoto } from '../../src/hooks/useClassifier';

const photo: RejectedPhoto = {
  id: 1,
  source: { kind: 'upload', name: 'cow.jpg' },
  displayUrl: 'blob:x',
  width: 640,
  height: 480,
  inputCheck: { score: 0.61, threshold: 0.85, passed: false, version: 'v' },
  file: new Blob(['x'], { type: 'image/jpeg' }),
};

describe('InputCheckNotice', () => {
  it('shows the heading, reason, tip and three actions, and focuses the heading', async () => {
    const onAnother = vi.fn();
    const onPhotoTips = vi.fn();
    const onAnalyseAnyway = vi.fn();
    render(
      <InputCheckNotice
        photo={photo}
        onAnother={onAnother}
        onPhotoTips={onPhotoTips}
        onAnalyseAnyway={onAnalyseAnyway}
      />,
    );
    const heading = screen.getByRole('heading', {
      name: "This doesn't look like a photo of poultry droppings.",
    });
    expect(heading).toHaveFocus();
    expect(screen.getByText(/only knows four kinds of chicken droppings/)).toBeInTheDocument();
    expect(screen.getByText(/close up from above/)).toBeInTheDocument();
    expect(screen.queryByTestId('verdict')).toBeNull();

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: /Try another photo/ }));
    await user.click(screen.getByRole('link', { name: 'Photo tips' }));
    await user.click(screen.getByRole('button', { name: 'Analyse anyway' }));
    expect(onAnother).toHaveBeenCalledOnce();
    expect(onPhotoTips).toHaveBeenCalledOnce();
    expect(onAnalyseAnyway).toHaveBeenCalledOnce();
  });

  it('reaches every action with the keyboard, in order', async () => {
    render(
      <InputCheckNotice photo={photo} onAnother={vi.fn()} onPhotoTips={vi.fn()} onAnalyseAnyway={vi.fn()} />,
    );
    const user = userEvent.setup();
    await user.tab();
    expect(screen.getByRole('button', { name: /Try another photo/ })).toHaveFocus();
    await user.tab();
    expect(screen.getByRole('link', { name: 'Photo tips' })).toHaveFocus();
    await user.tab();
    expect(screen.getByRole('button', { name: 'Analyse anyway' })).toHaveFocus();
  });
});
