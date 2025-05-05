import time
import shutil
from qtpy.QtCore import QObject
from qtpy.QtWidgets import QMessageBox
from napari.qt import create_worker
from napari.utils.notifications import show_info, show_warning, show_error

class YoloHandler(QObject):
    def __init__(self, parent_widget, yolo_octron):
        super().__init__()
        self.w = parent_widget
        self.yolo = yolo_octron
        self.trained_models = {}

    def connect_signals(self):
        w = self.w
        w.generate_training_data_btn.clicked.connect(self.init_training_data_threaded)
        w.start_stop_training_btn.clicked.connect(self.init_yolo_training_threaded)
        w.predict_start_btn.clicked.connect(self.init_yolo_prediction_threaded)
        w.predict_iou_thresh_spinbox.valueChanged.connect(self.on_iou_thresh_change)
        
    def _uncouple_worker_polygons(self):
        try:
            w = self.w
            w.polygon_worker.yielded.disconnect(self._polygon_yielded)
            w.polygon_worker.finished.disconnect(self._on_polygon_finished)
            w.polygon_worker.quit()
        except Exception as e:
            print(f"Error when uncoupling polygon worker: {e}")
            
    def _uncouple_worker_training_data(self):
        try:
            w = self.w
            w.training_data_worker.yielded.disconnect(self._training_data_yielded)
            w.training_data_worker.finished.disconnect(self._on_training_data_finished)
            w.training_data_worker.quit()
        except Exception as e:
            print(f"Error when uncoupling training data worker: {e}")   

    def _training_data_export(self):
        w = self.w
        """
        MAIN MANAGER FOR TRAINING DATA EXPORT
        """
        if not w.training_data_interrupt and w.training_data_generated:
            self._on_training_data_finished()
            return

        if not hasattr(w, 'training_data_worker'):
            self._create_worker_training_data()
            w.generate_training_data_btn.setStyleSheet('QPushButton { color: #e7a881;}')
            w.generate_training_data_btn.setText('🅧 Interrupt')
            w.training_data_worker.start()
            w.training_data_interrupt = False

        elif not w.training_data_worker.is_running:
            self._uncouple_worker_training_data()
            self._create_worker_training_data()
            w.generate_training_data_btn.setStyleSheet('QPushButton { color: #e7a881;}')
            w.generate_training_data_btn.setText('🅧 Interrupt')
            w.training_data_worker.start()
            w.training_data_interrupt = False

        else:
            w.training_data_worker.quit()
            w.generate_training_data_btn.setStyleSheet('QPushButton { color: #8ed634;}')
            w.generate_training_data_btn.setText('▷ Generate')
            w.training_data_interrupt = True
            w.polygons_generated = False

        w.training_data_folder_label.setEnabled(True)
        w.training_data_folder_label.setText(f'→{self.yolo.training_path.as_posix()[-38:]}')


    def _polygon_generation(self):
        w = self.w
        """
        MAIN MANAGER FOR POLYGON GENERATION
        polygon_worker()
        Manages thread worker for generating polygons
        """
        # Check if the worker has already run and was not interrupted.
        # If so, do not create a new worker, but just call the callback function.
        if not self.polygon_interrupt and self.polygons_generated:
            self._on_polygon_finished()
            return
        # Otherwise, create a new worker and manage interruptions
        if not hasattr(self, 'polygon_worker'):
            self._create_worker_polygons()
            w.generate_training_data_btn.setStyleSheet('QPushButton { color: #e7a881;}')
            w.generate_training_data_btn.setText(f'🅧 Interrupt')
            w.polygon_worker.start()
            w.polygon_interrupt = False
        elif hasattr(self, 'polygon_worker') and not w.polygon_worker.is_running:
            # Worker exists but is not running - clean up and create a new one
            self._uncouple_worker_polygons()
            self._create_worker_polygons()
            w.generate_training_data_btn.setStyleSheet('QPushButton { color: #e7a881;}')
            w.generate_training_data_btn.setText(f'🅧 Interrupt')
            w.polygon_worker.start()
            self.polygon_interrupt = False
        elif hasattr(self, 'polygon_worker') and w.polygon_worker.is_running:
            w.polygon_worker.quit()
            w.generate_training_data_btn.setStyleSheet('QPushButton { color: #8ed634;}')
            w.generate_training_data_btn.setText(f'▷ Generate')
            self.polygon_interrupt = True
            

    def init_training_data_threaded(self):
        """
        This function manages the creation of polygon data and the subsequent
        creation / the export of training data.
        It kicks off the pipeline by calling the polygon generation function.
        
        """
        # Whenever the button "Generate" is clicked, 
        # the training data generation pipeline is started anew.
        if not hasattr(self, 'polygon_worker') and not hasattr(self, 'training_data_worker'):   
            self.polygons_generated = False
            self.training_data_generated = False   
        # Sanity check 
        if not self.w.project_path:
            show_warning("Please select a project directory first.")
            return
        if not self.yolo:
            show_warning("Please load a YOLO first.")
            return
        # Check status of "Prune" checkbox
        prune = self.w.train_prune_checkBox.isChecked()
        # Check whether training folder should be overwritten or not 
        self.yolo.clean_training_dir = self.w.train_data_overwrite_checkBox.isChecked()
        # Set the project_path (which also takes care of setting up training subfolders)
        if not self.yolo.project_path:
            self.yolo.project_path = self.w.project_path
        elif self.yolo.project_path != self.w.project_path: 
            # Assuming that the user wants to change the project path
            self.yolo.project_path = self.w.project_path

        self.w.save_object_organizer() # This is safe, since it checks whether a video was loaded
        
        # TODO: Could make `prepare_labels` async as well ... 
        try:
            # After saving the object organizer, extract info from all 
            # available .json files in the project directory
            self.yolo.prepare_labels(
                        prune_empty_labels=prune, 
                        min_num_frames=5, # hardcoded ... less than 5 frames are not useful
                        verbose=True, 
            )
        except AssertionError as e:
            print(f"😵 Error when preparing labels: {e}")
            return
        
        if not self.yolo.clean_training_dir:
            # Check if the training folder already exists
            # If it does, we can skip everything after this step
            
            if self.yolo.data_path is not None and self.yolo.data_path.exists():
                # Remove any model subdirectories
                # Assuming /training as the model subfolder which is set during YOLO training initialization
                assert self.yolo.training_path is not None 
                if self.yolo.training_path / 'training' in self.yolo.training_path.glob('*'):
                    shutil.rmtree(self.yolo.training_path / 'training')
                    print(f"Removed existing model subdirectory '{self.yolo.training_path / 'training'}'")
                # TODO: Since we just generated the labels_dict (in prepare_labels above), 
                # a rudimentary check is actually possible, comparing the total number of expected labeled 
                # frames and the number of images in the training folder. I am skipping any checks for now.
                # Show a warning dialog that user must dismiss
                warning_dialog = QMessageBox()
                warning_dialog.setIcon(QMessageBox.Warning)
                warning_dialog.setWindowTitle("Existing Training Data")
                warning_dialog.setText("Training data directory already exists.")
                warning_dialog.setInformativeText("The existing training data will be used without regeneration. "
                                                  "No checks are performed on the training data folder. "
                                                  "If you want to regenerate the data, please check the 'Overwrite' option.")
                warning_dialog.setStandardButtons(QMessageBox.Ok)
                warning_dialog.exec_()
                self.polygons_generated = True
                self.training_data_generated = True
                self._on_training_data_finished()
                
                print(f"Training data path '{self.yolo.data_path.as_posix()}' already exists. Using existing directory.")
                return

        # Else ... continue the training data generation pipeline
        # Kick off polygon generation - check are done within the following functions 
        self._polygon_generation()
        return
            

    def _create_worker_polygons(self):
        w = self.w
        # Create a new worker for polygon generation
        # Watershed? 
        enable_watershed = self.w.train_data_watershed_checkBox.isChecked()
        self.yolo.enable_watershed = enable_watershed
        w.polygon_worker = create_worker(self.yolo.prepare_polygons)
        w.polygon_worker.setAutoDelete(True) # auto destruct !!
        w.polygon_worker.yielded.connect(self._polygon_yielded)
        w.polygon_worker.finished.connect(self._on_polygon_finished)
        self.w.train_polygons_overall_progressbar.setEnabled(True)    
        self.w.train_polygons_frames_progressbar.setEnabled(True)
        self.w.train_polygons_label.setEnabled(True)

    def _polygon_yielded(self, value):
        """
        polygon_worker()
        Called upon yielding from the batch polygon generation thread worker.
        Updates the progress bar and label text next to it.
        """
        no_entry, total_label_dict, label, frame_no, total_frames = value
        self.w.train_polygons_overall_progressbar.setMaximum(total_label_dict)
        self.w.train_polygons_overall_progressbar.setValue(no_entry-1)
        self.w.train_polygons_frames_progressbar.setMaximum(total_frames) 
        self.w.train_polygons_frames_progressbar.setValue(frame_no)   
        self.w.train_polygons_label.setText(label)  

    def _on_polygon_finished(self):
        """
        polygon_worker()
        Callback for when polygon generation worker has finished executing. 
        """

        self.w.train_polygons_overall_progressbar.setValue(0)
        self.w.train_polygons_frames_progressbar.setValue(0)
        self.w.train_polygons_overall_progressbar.setEnabled(False)    
        self.w.train_polygons_frames_progressbar.setEnabled(False)
        self.w.train_polygons_label.setText('')
        self.w.train_polygons_label.setEnabled(False) 
        
        if self.w.polygon_interrupt:
            show_warning("Polygon generation interrupted.")  
            self.w.polygons_generated = False
            self.w.generate_training_data_btn.setStyleSheet('QPushButton { color: #8ed634;}')
            self.w.generate_training_data_btn.setText(f'▷ Generate')
        else:
            self.w.polygons_generated = True
            
        # If self.w.polygons_generated is True, then start the 
        # training data export worker right after ...
        if self.w.polygons_generated and not self.w.training_data_generated:
            # split train/val/test then kick off data export
            self.yolo.prepare_split()
            self._training_data_export()
        else:
            pass

    def _create_worker_training_data(self):
        w = self.w
        # Create a new worker for training data generation / export
        w.training_data_worker = create_worker(self.yolo.create_training_data)
        w.training_data_worker.setAutoDelete(True) # auto destruct !!
        w.training_data_worker.yielded.connect(self._training_data_yielded)
        w.training_data_worker.finished.connect(self._on_training_data_finished)
        self.w.train_export_overall_progressbar.setEnabled(True)    
        self.w.train_export_frames_progressbar.setEnabled(True)
        self.w.train_polygons_label.setEnabled(True)

            
    def _training_data_yielded(self, value):
        """
        training_data_worker()
        Called upon yielding from the batch training data generation thread worker.
        Updates the progress bar and label text next to it.
        """
        no_entry, total_label_dict, label, split, frame_no, total_frames = value
        self.w.train_export_overall_progressbar.setMaximum(total_label_dict)
        self.w.train_export_overall_progressbar.setValue(no_entry-1)
        self.w.train_export_frames_progressbar.setMaximum(total_frames) 
        self.w.train_export_frames_progressbar.setValue(frame_no)   
        self.w.train_export_label.setText(f'{label} ({split})')   

    
    def _on_training_data_finished(self):
        """
        training_data_worker()
        Callback for when training data generation worker has finished executing. 
        """
        self.w.generate_training_data_btn.setStyleSheet('QPushButton { color: #8ed634;}')
        self.w.generate_training_data_btn.setText(f'▷ Generate')
        self.w.train_export_overall_progressbar.setValue(0)
        self.w.train_export_frames_progressbar.setValue(0)
        self.w.train_export_overall_progressbar.setEnabled(False)    
        self.w.train_export_frames_progressbar.setEnabled(False)
        self.w.train_export_label.setText('')
        self.w.train_export_label.setEnabled(False) 
        
        if self.w.training_data_interrupt:
            show_warning("Training data generation interrupted.")  
            self.w.training_data_generated = False
        else:
            show_info("Training data generation finished.")
            self.w.training_data_generated = True
            self.w.generate_training_data_btn.setText(f'✓ Done.')
            self.w.generate_training_data_btn.setEnabled(False)   
            self.w.train_data_overwrite_checkBox.setEnabled(False)
            self.w.train_prune_checkBox.setEnabled(False)
            self.w.train_data_watershed_checkBox.setEnabled(False)
            self.yolo.write_yolo_config()
            # Enable next part (YOLO training) of the pipeline 
            self.w.train_train_groupbox.setEnabled(True)
            self.w.launch_tensorboard_checkBox.setEnabled(False)
            self.w.start_stop_training_btn.setStyleSheet('QPushButton { color: #8ed634;}')
            self.w.start_stop_training_btn.setText(f'▷ Train')
        

    def _update_training_progress(self, progress_info):
        """
        Handle training progress updates from the worker thread.
        When finished, enable the next step.
        
        Parameters
        ----------
        progress_info : dict
            Dictionary containing training progress information
        """
        current_epoch = progress_info['epoch']
        total_epochs = progress_info['total_epochs']
        epoch_time = progress_info['epoch_time']
        remaining_time = progress_info['remaining_time']
        finish_time = progress_info['finish_time']
        
        # Format the finish time (without year as requested)
        finish_time_str = ' '.join(time.ctime(finish_time).split()[:-1])
   
        self.w.train_epochs_progressbar.setMaximum(total_epochs)
        self.w.train_epochs_progressbar.setValue(current_epoch)        
        self.w.train_finishtime_label.setText(f'↬ {finish_time_str}')
        
        print(f"Epoch {current_epoch}/{total_epochs} - Time for epoch: {epoch_time:.1f}s")
        print(f"Estimated time remaining: {remaining_time:.1f} seconds")    
        print(f"Estimated finish time: {finish_time_str}")  

        if current_epoch == total_epochs: 
            self.w.training_finished = True
            self.w.start_stop_training_btn.setStyleSheet('QPushButton { color: #8ed634;}')
            self.w.start_stop_training_btn.setText(f'✓ Done.')
            self.w.train_epochs_progressbar.setEnabled(False)  
            self.w.train_finishtime_label.setEnabled(False)
            # Enable the prediction tab
            self.w.toolBox.widget(3).setEnabled(True) # Prediction
            self.w.predict_video_drop_groupbox.setEnabled(True)
            self.w.predict_video_predict_groupbox.setEnabled(True)
            
    
    def _yolo_trainer(self):
        if not self.w.device_label:
            show_error("No device label found for YOLO.")
            return
        else:
            show_info(f"Training on device: {self.w.device_label}")
        
        # Call the training function which yields progress info
        for progress_info in self.yolo.train(
                                            device=self.w.device_label, 
                                            imagesz=self.w.image_size_yolo,
                                            epochs=self.w.num_epochs_yolo,
                                            save_period=self.w.save_period,
                                        ):
            # Yield the progress info back to the GUI thread
            yield progress_info
            
            
    def _create_yolo_trainer(self):
        # Create a new worker for YOLO training 
        w = self.w
        w.yolo_trainer_worker = create_worker(self._yolo_trainer)
        w.yolo_trainer_worker.setAutoDelete(True)  # auto destruct !!
        w.yolo_trainer_worker.yielded.connect(self._update_training_progress)
        self.w.train_epochs_progressbar.setEnabled(True)    
        self.w.train_finishtime_label.setEnabled(True)
        self.w.train_finishtime_label.setText('↬ ... wait one epoch')    
        if self.w.launch_tensorbrd:
            self.w.yolo.quit_tensorboard()
            self.w.yolo.launch_tensorboard()
            

    def init_yolo_training_threaded(self):
        """
        This function manages the training of the YOLO model.
        """
        if self.w.training_finished:
            return
        
        # Sanity check 
        if not self.w.project_path:
            show_warning("Please select a project directory first.")
            return
        if not hasattr(self, 'yolo_octron'):
            show_warning("Please load YOLO first.")
            return
        
        index_model_list = self.w.yolomodel_list.currentIndex()
        if index_model_list == 0:
            show_warning("Please select a YOLO model")
            return
        model_name = self.w.yolomodel_list.currentText()
        # Reverse lookup model_id
        for model_id, model in self.w.yolomodels_dict.items():
            if model['name'] == model_name:
                break        
        index_imagesize_list = self.w.yoloimagesize_list.currentIndex()
        if index_imagesize_list == 0:
            show_warning("Please select an image size")
            return 
        self.w.image_size_yolo = int(self.w.yoloimagesize_list.currentText())                                     
        # Check status of "Launch Tensorboard" checkbox
        self.w.launch_tensorbrd = False #self.w.launch_tensorboard_checkBox.isChecked()   
        # TODO: Implement these options
        #resume_training = self.w.train_resume_checkBox.isChecked()    
        #overwrite = self.w.train_training_overwrite_checkBox.isChecked()  
        
        self.w.num_epochs_yolo = int(self.w.num_epochs_input.value())   
        if self.w.num_epochs_yolo <= 1:
            show_warning("Please select a number of epochs >1")
            return
        self.w.save_period = int(self.w.save_period_input.value())
        
        # LOAD YOLO MODEL 
        print(f"Loading YOLO model {model_id}")
        yolo_model = self.w.yolo.load_model(model_id)
        if not yolo_model:
            show_warning("Could not load YOLO model.")
            return
        
        # Deactivate the training data generation box 
        self.w.train_generate_groupbox.setEnabled(False)
        # Otherwise, create a new worker and manage interruptions
        if not hasattr(self, 'yolo_trainer_worker'):
            self._create_yolo_trainer()
            self.w.start_stop_training_btn.setStyleSheet('QPushButton { color: #e7a881;}')
            self.w.start_stop_training_btn.setText(f'↯ Training')
            self.w.yolo_trainer_worker.start()
            self.w.start_stop_training_btn.setEnabled(False)
            # Disable the training data generation box
            self.w.toolBox.widget(1).setEnabled(False) # Annotation
            
    ###### YOLO PREDICTION ###########################################################################
    
    def on_iou_thresh_change(self, value):
        """
        Callback for self.w.predict_iou_thresh_spinbox.
        If IOU threshold is < 0.01, 
        check and disable the 'single_subject_checkBox'.
        This is because at IOU < 0.01, only one object will be tracked
        by fusing all detections into 1 -> so it has the same effect. 
        
        """
        if value < 0.01:
            self.w.single_subject_checkBox.setChecked(True)
            self.w.single_subject_checkBox.setEnabled(False)
        else:
            self.w.single_subject_checkBox.setEnabled(True)
            self.w.single_subject_checkBox.setChecked(False)
            
        
    def _update_prediction_progress(self, progress_info):
        """
        Handle prediction progress updates from the worker thread.
        Updates progress bars and displays timing information.
        
        Parameters
        ----------
        progress_info : dict
            Dictionary containing prediction progress information
        """
        stage = progress_info.get('stage', '')
        
        if stage == 'processing':
            # Update UI for video processing
            video_name = progress_info.get('video_name', '')
            video_index = progress_info.get('video_index', 0)
            total_videos = progress_info.get('total_videos', 1)
            frame = progress_info.get('frame', 0)
            total_frames = progress_info.get('total_frames', 1)
            frame_time = progress_info.get('frame_time', 0) 
            
            remaining_time = (total_frames * frame_time) - (frame * frame_time)
            finish_time = time.time() + remaining_time
            finish_time_str = ' '.join(time.ctime(finish_time).split()[:-1])

            # Update labels
            if len(video_name) > 21:
                prefix = '...'
            else:
                prefix = ''
            shortened_video_name = f'{prefix}{video_name[-21:]}'
            
            self.w.predict_current_videoname_label.setText(f"{shortened_video_name}")
            self.w.predict_finish_time_label.setText(f"{frame_time:.2f}s per frame | Video completes ~ {finish_time_str}")
            
            # Update progress bars
            self.w.predict_overall_progressbar.setMaximum(total_videos)
            self.w.predict_current_video_progressbar.setMaximum(total_frames)
            self.w.predict_overall_progressbar.setValue(video_index)
            self.w.predict_current_video_progressbar.setValue(frame)
            
        elif stage == 'video_complete':
            # Show results? 
            save_dir = progress_info.get('save_dir', '')
            if self.w.view_prediction_resuts: 
                for label, track_id, _, _, _  in self.w.yolo.show_predictions(save_dir=save_dir):
                    print(f"Adding tracking result to viewer | Label: {label}, Track ID: {track_id}")     
            
        elif stage == 'complete':
            self.w.yolo_prediction_worker.quit()
            # Reset video progress bar for next video
            self.w.predict_current_video_progressbar.setValue(0)
            self.w.predict_overall_progressbar.setValue(0)
            self.w.predict_current_video_progressbar.setEnabled(False)
            self.w.predict_overall_progressbar.setEnabled(False)
            
            
            self.w.predict_current_videoname_label.setText('')
            self.w.predict_finish_time_label.setText('')
            self.w.predict_current_videoname_label.setEnabled(False)
            self.w.predict_finish_time_label.setEnabled(False)
            
            # Re-enable UI elements
            self.w.predict_start_btn.setStyleSheet('QPushButton { color: #8ed634;}')
            self.w.predict_start_btn.setText('▷ Predict')
            self.w.predict_start_btn.setEnabled(True)
            self.w.toolBox.widget(1).setEnabled(True)  # Re-enable Annotation tab
            self.w.toolBox.widget(2).setEnabled(True)  # Re-enable Training tab
            self.w.predict_video_drop_groupbox.setEnabled(True)
                
            
    
    def _yolo_predictor(self):
        if not self.w.device_label:
            show_error("No device label found for YOLO.")
            return
        else:
            show_info(f"Predicting on device: {self.w.device_label}")
        
        one_object_per_label = self.w.single_subject_checkBox.isChecked()
        # Call the training function which yields progress info
        for progress_info in self.w.yolo.predict_batch(
                                            videos_dict=self.w.videos_to_predict,
                                            model_path=self.w.model_predict_path,
                                            device=self.w.device_label,
                                            tracker_name=self.w.yolo_tracker_name,
                                            one_object_per_label=one_object_per_label,
                                            iou_thresh=self.w.iou_thresh,
                                            conf_thresh=self.w.conf_thresh,
                                            polygon_sigma=self.w.polygon_sigma,
                                            overwrite=self.w.overwrite_predictions, 
                                        ):

            # Yield the progress info back to the GUI thread
            yield progress_info
            
            
    def _create_yolo_predictor(self):
        # Create a new worker for YOLO prediction 
        w = self.w
        w.yolo_prediction_worker = create_worker(self._yolo_predictor)
        w.yolo_prediction_worker.setAutoDelete(True)  # auto destruct !!
        w.yolo_prediction_worker.yielded.connect(self._update_prediction_progress)
        self.w.predict_overall_progressbar.setEnabled(True)  
        self.w.predict_current_video_progressbar.setEnabled(True)   
        self.w.predict_current_videoname_label.setEnabled(True)
        self.w.train_finishtime_label.setText('↬ ... waiting for estimate')    
        self.w.predict_finish_time_label.setEnabled(True)


    def init_yolo_prediction_threaded(self):
        """
        This function manages the prediction of videos
        with custom trained YOLO models
        
        """
        
        # Sanity check 
        if not self.w.project_path:
            show_warning("Please select a project directory first.")
            return
        if not hasattr(self, 'yolo_octron'):
            show_warning("Please load YOLO first.")
            return
        
        index_model_list = self.w.yolomodel_trained_list.currentIndex()
        if index_model_list == 0:
            show_warning("Please select a YOLO model")
            return
        model_name = self.w.yolomodel_trained_list.currentText()
        # The self.w.trained_models dictionary contains the model name as last 5 folder names
        # in the project path as key, and the model path as value
        assert model_name in self.w.trained_models, \
            f"Model {model_name} not found in trained models: {self.w.trained_models}"
        self.w.model_predict_path = self.w.trained_models[model_name]
        # Tracker
        index_tracker_list = self.w.yolomodel_tracker_list.currentIndex()
        if index_tracker_list == 0:
            show_warning("Please select a tracker")
            return 
        # Check if there are any videos to predict 
        if not self.w.videos_to_predict:
            show_warning("Please select a video to predict.")
            return
        
        self.w.yolo_tracker_name = self.w.yolomodel_tracker_list.currentText()                              
        # Check status of "view results" checkbox
        self.w.view_prediction_resuts = self.w.open_when_finish_checkBox.isChecked()   
        self.w.polygon_sigma = float(self.w.predict_polygo_sigma_spinbox.value())
        self.w.conf_thresh = float(self.w.predict_conf_thresh_spinbox.value())
        self.w.iou_thresh = float(self.w.predict_iou_thresh_spinbox.value())
        self.w.overwrite_predictions = self.w.overwrite_prediction_checkBox.isChecked()    
        
        # Deactivate the training data generation box 
        self.w.train_generate_groupbox.setEnabled(False)
        # Create new prediction worker
        self._create_yolo_predictor()
        self.w.predict_start_btn.setStyleSheet('QPushButton { color: #e7a881;}')
        self.w.predict_start_btn.setText(f'↯ Predicting')
        self.w.yolo_prediction_worker.start()
        self.w.predict_start_btn.setEnabled(False)
        # Disable the annotation + training data generation tabs
        self.w.toolBox.widget(1).setEnabled(False) # Annotation
        self.w.toolBox.widget(2).setEnabled(False) # Training
        # And the video dropbox
        self.w.predict_video_drop_groupbox.setEnabled(False)
        
    
    def refresh_trained_model_list(self):
        """
        Refresh the trained model list combobox with the current models in the project directory
        """
        # Clear the old list, and re-instantiate
        self.w.yolomodel_trained_list.clear()
        self.w.yolomodel_trained_list.addItem('Choose model ...')
        
        trained_models = self.w.yolo.find_trained_models(search_path=self.w.project_path)
        if not trained_models:
            self.w.toolBox.widget(3).setEnabled(False)
            return
        
        # Write the trained models to yolomodel_trained_list one by one
        for model in trained_models:
            # This is to clearly identify the model
            # in the list, since the model name is not unique
            model_name = '/'.join(model.parts[-5:])
            if model_name not in self.w.trained_models:
                self.w.trained_models[model_name] = model
            self.w.yolomodel_trained_list.addItem(model_name)
        # Enable prediction tab if trained models are available
        self.w.toolBox.widget(3).setEnabled(True)
        self.w.predict_video_drop_groupbox.setEnabled(True)
        self.w.predict_video_predict_groupbox.setEnabled(True)
        self.w.predict_start_btn.setEnabled(True)
        self.w.predict_start_btn.setStyleSheet('QPushButton { color: #8ed634;}')
        self.w.predict_start_btn.setText(f'▷ Predict')