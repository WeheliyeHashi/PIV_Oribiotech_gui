#%%
from tkinter import Tk, Label, Entry, Button, Checkbutton, IntVar, StringVar, filedialog
import os
import subprocess
#%%
class PIVProcessorGUI:
    def __init__(self, master):
        self.master = master
        master.title("PIV Processor GUI")
        master.geometry("800x400")  # Set the size of the window

        self.base_filepath = StringVar()
        self.model_path = StringVar()
        self.n_cpus = IntVar(value=4)
        self.plot_video = IntVar(value=1)
        self.normalize_data = IntVar(value=1)
        self.use_metadata_file = IntVar(value=0)
        self.metadata_file = StringVar()
        self.status_label = Label(master, text="")
        self.status_label.grid(row=8, columnspan=3, pady=10)

        Label(master, text="Base File Path:").grid(row=0, column=0, sticky='w', padx=10, pady=5)
        Entry(master, textvariable=self.base_filepath, width=70).grid(row=0, column=1, padx=10, pady=5)
        Button(master, text="Browse", command=self.browse_base).grid(row=0, column=2, padx=10, pady=5)

        Label(master, text="Model Path:").grid(row=1, column=0, sticky='w', padx=10, pady=5)
        Entry(master, textvariable=self.model_path, width=70).grid(row=1, column=1, padx=10, pady=5)
        Button(master, text="Browse", command=self.browse_model).grid(row=1, column=2, padx=10, pady=5)

        Label(master, text="Number of CPUs:").grid(row=2, column=0)
        Entry(master, textvariable=self.n_cpus).grid(row=2, column=1)

        Checkbutton(master, text="Save images", variable=self.plot_video).grid(row=3, columnspan=2)
        Checkbutton(master, text="Normalize Data", variable=self.normalize_data).grid(row=4, columnspan=2)
        Checkbutton(master, text="Use Metadata File", variable=self.use_metadata_file, command=self.toggle_metadata_file).grid(row=5, columnspan=2)

        self.metadata_button = Button(master, text="Browse Metadata File", command=self.browse_metadata, state='disabled')
        self.metadata_button.grid(row=6, column=0, columnspan=3)

        Button(master, text="Run Processing", command=self.run_processing).grid(row=7, columnspan=3)

    def browse_base(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.base_filepath.set(folder_selected)

    def browse_model(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.model_path.set(folder_selected)

    def browse_metadata(self):
        file_selected = filedialog.askopenfilename()
        if file_selected:
            self.metadata_file.set(file_selected)

    def toggle_metadata_file(self):
        if self.use_metadata_file.get():
            self.metadata_button.config(state='normal')
        else:
            self.metadata_button.config(state='disabled')
            self.metadata_file.set('')

    def run_processing(self):

        self.status_label.config(text="Analysis in process...")
        self.master.update_idletasks()  # Update the GUI to show the message
        base_filepath = self.base_filepath.get()
        model_path = self.model_path.get()
        n_cpus = self.n_cpus.get()
        plot_video = bool(self.plot_video.get())
        normalize_data = bool(self.normalize_data.get())
        metadata_file = None if not self.use_metadata_file.get() else self.metadata_file.get()

        command = [
            'PIV_main_processing',
            '--base_filepath', base_filepath,
            '--model_path', model_path,
            '--n_cpus', str(n_cpus),
            '--plot_video', str(plot_video),
            '--normalize_data', str(normalize_data),
        ]
        
        if metadata_file:
            command.extend(['--metadata_file', metadata_file])

        subprocess.run(command)
        self.status_label.config(text="Analysis completed")

def main():
    root = Tk()
    gui = PIVProcessorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
# %%
