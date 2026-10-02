import { CalibrationWizardContent } from '../calibration/calibration-wizard';
import { InlineAlert } from '../shared/inline-alert';

/**
 * reBot B601-RS setup: the shared zero-pose calibration wizard with RS guidance.
 * The plugin turns motor torque off before zeroing, so the arm goes limp.
 */
export const ReBotRSSetupWizardContent = () => (
    <CalibrationWizardContent
        title='Move the reBot B601-RS to its zero pose'
        tips={
            <InlineAlert variant='warning'>
                Motor torque is off during calibration, so the arm cannot hold itself up. Support it while you move it.
                Make sure the CAN interface is up at 1 Mbit/s before you begin.
            </InlineAlert>
        }
    />
);
