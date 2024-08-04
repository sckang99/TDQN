# %%
import os
import glob
import traceback
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

runs_path = 'runs'

class extract_TensorBoard:
    # Extraction function
    def __init__(self, runs_path = 'runs'):
        self.runs_path = runs_path


    def tflog2pandas(self, path):
        runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
        try:
            event_acc = EventAccumulator(path)
            event_acc.Reload()
            tags = event_acc.Tags()["scalars"]
            for tag in tags:
                event_list = event_acc.Scalars(tag)
                values = list(map(lambda x: x.value, event_list))
                step = list(map(lambda x: x.step, event_list))
                r = {"metric": [tag] * len(step), "value": values, "step": step}
                r = pd.DataFrame(r)
                runlog_data = pd.concat([runlog_data, r])
        # Dirty catch of DataLossError
        except Exception:
            print("Event file possibly corrupt: {}".format(path))
            traceback.print_exc()
        return runlog_data

    # Extraction function
    
    def extract_and_save(self):
        all_folders = [d for d in glob.glob(os.path.join(self.runs_path,'*')) if os.path.isdir(d)]
        if all_folders:
            path = max(all_folders, key=os.path.getmtime)
            print(path)
            df = self.tflog2pandas(path)
            tf = (df["metric"] == "Testing performance (Sharpe Ratio)")
            df2 = df.loc[tf,("step", "value")]
            df2 = df2.rename(columns={"value": "Testing"})

            tf = (df["metric"] == "Training performance (Sharpe Ratio)")
            df3 = df.loc[tf,("step", "value")]
            df3 = df3.rename(columns={"value": "Training"})

            df = pd.concat([df2, df3], axis=1)

            filename = path + "\\" + "SR.csv"
            df.to_csv(filename)
