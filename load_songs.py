import midi
import os
import util 
import numpy as np
import traceback

patterns = {}
main_dir = 'songs'
all_samples = []
all_lens = []
print("Loading songs")
for root, dirs, files in os.walk(main_dir):
    for filename in files:
        path = os.path.join(root, filename)
        if not (path.endswith('.mid') or path.endswith('.MID') or path.endswith('.midi') or path.endswith('.MIDI')):
            continue
        try:
            samples = midi.midi_to_samples(path)
        except Exception as e:
            traceback.print_exc()
            #print("Error path: {} does not support conversion".format(path))
            continue
        if len(samples) < 8:
            continue

        samples, lens = util.generate_add_centered_transpose(samples)
        all_samples.extend(samples)
        all_lens.extend(lens)

assert(sum(all_lens) == len(all_samples))
print("Saving {} samples".format(len(all_samples)))
all_samples = np.array(all_samples, dtype=np.uint8)
all_lens = np.array(all_lens, dtype=np.uint32)
np.save('samples.npy', all_samples)
np.save('lengths.npy', all_lens)
print("Done")
