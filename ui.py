#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Code to make a GUI to run the RiceBraille project using wxPython.

Code is adapted from Jan Bodnar's ZetCode wxPython tutorial.
sample code author: Jan Bodnar
website: www.zetcode.com
last modified: April 2018
"""

import wx
import re
import os
from VideoTracker import VideoTracker
from datetime import datetime

# inherits from wx.Frame, a standard container widget
class BrailleGUI(wx.Frame):

    def __init__(self, parent, title):
        # initialize window params
        super(BrailleGUI, self).__init__(parent, title=title, size=(500, 600))
        self.auto_finger = 0
        self.auto_page = 0
        self.char_track = 0
        self.show_frames = 0
        self.page_length = 0
        self.page_width = 0
        self.brfFile = "N/A" 
        self.videoFile = "" 
        self.fingers = [0] * 8
        self.dirPath = ":C/User/Documents/Braille_Outputs"

        self.InitUI()
        self.Centre()

    def InitUI(self):
        
        panel = wx.Panel(self)

        # obtain system font
        font = wx.SystemSettings.GetFont(wx.SYS_SYSTEM_FONT)
        font.SetPointSize(9)

        # we have one vertical sizer, and then a bunch of horizontal sizers inside that vertical sizer
        vbox = wx.BoxSizer(wx.VERTICAL)

        ### SECTION 1: CORE AUTOMATIC FEATURES
        # Instructions
        hbox0 = wx.BoxSizer(wx.HORIZONTAL)
        st0 = wx.StaticText(panel, label='Select core automatic features to enable:')
        st0.SetFont(font)
        hbox0.Add(st0, flag=wx.LEFT|wx.RIGHT|wx.TOP, border=8)
        vbox.Add(hbox0)

        # Automatic Finger Calibration
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        st1 = wx.StaticText(panel, label='Enable Automatic Finger Calibration:')
        st1.SetFont(font)
        hbox1.Add(st1, flag=wx.LEFT|wx.RIGHT|wx.TOP, border=8)
        self.auto_finger = wx.CheckBox(panel)
        hbox1.Add(self.auto_finger, flag=wx.LEFT|wx.RIGHT|wx.TOP, border=8, proportion=1)
        vbox.Add(hbox1)

        # box.Add can insert not only widgets, but also empty space
        vbox.Add((-1, 5))

        # Automatic Page Calibration
        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        st2 = wx.StaticText(panel, label='Enable Automatic Page Calibration:')
        st2.SetFont(font)
        hbox2.Add(st2, flag=wx.LEFT|wx.RIGHT|wx.TOP, border=8)
        self.auto_page = wx.CheckBox(panel)
        self.auto_page.SetValue(True)
        hbox2.Add(self.auto_page, flag=wx.LEFT|wx.RIGHT|wx.TOP, border=8, proportion=1)
        vbox.Add(hbox2)

        vbox.Add((-1, 25))

        ### SECTION 2: SUPPLEMENTARY FEATURES
        # Instructions
        hbox3 = wx.BoxSizer(wx.HORIZONTAL)
        st3 = wx.StaticText(panel, label='If character tracking is desired, upload a .brf file:')
        st3.SetFont(font)
        hbox3.Add(st3, flag=wx.LEFT|wx.RIGHT|wx.TOP, border=8)
        vbox.Add(hbox3)

        # Character Tracking
        hbox4 = wx.BoxSizer(wx.HORIZONTAL)
        st4 = wx.StaticText(panel, label='Upload .brf file:')
        st4.SetFont(font)
        hbox4.Add(st4, flag=wx.LEFT|wx.RIGHT|wx.TOP, border=8)
        btn0 = wx.Button(panel, label='Choose file', size=(70, 30))
        hbox4.Add(btn0)
        self.Bind(wx.EVT_BUTTON, self.OnOpenBrf, btn0)
        vbox.Add(hbox4)

        # Indication if file has been added
        hbox_brf = wx.BoxSizer(wx.HORIZONTAL)
        self.st_brf = wx.StaticText(panel, label="Uploaded File: " + self.brfFile)
        self.st_brf.SetFont(font)
        hbox_brf.Add(self.st_brf, flag=wx.LEFT|wx.RIGHT|wx.TOP, border=8)
        vbox.Add(hbox_brf)

        vbox.Add((-1, 25))

        ### SECTION 3: Additional Options
        # Instructions
        hbox5 = wx.BoxSizer(wx.HORIZONTAL)
        st5 = wx.StaticText(panel, label='Select additional options to enable:')
        st5.SetFont(font)
        hbox5.Add(st5, flag=wx.LEFT|wx.RIGHT|wx.TOP, border=8)
        vbox.Add(hbox5)

        # Show frames (ie. user gets to see video as it is processed frame by frame)
        hbox6 = wx.BoxSizer(wx.HORIZONTAL)
        st6 = wx.StaticText(panel, label='Show frames:')
        st6.SetFont(font)
        hbox6.Add(st6, flag=wx.LEFT|wx.RIGHT|wx.TOP, border=8)
        self.show_frames = wx.CheckBox(panel)
        self.show_frames.SetValue(True)
        hbox6.Add(self.show_frames, flag=wx.LEFT|wx.RIGHT|wx.TOP, border=8, proportion=1)
        vbox.Add(hbox6)

        vbox.Add((-1, 5))

        # Select fingers to track
        hbox7 = wx.BoxSizer(wx.HORIZONTAL)
        st7 = wx.StaticText(panel, label='Select fingers to track:')
        st7.SetFont(font)
        hbox7.Add(st7, flag=wx.LEFT|wx.RIGHT|wx.TOP, border=8)
        finger_labels = ['p', 'r', 'm', 'i', 'i', 'm', 'r', 'p']
        for i in range(8):
            self.fingers[i] = wx.CheckBox(panel, label=finger_labels[i])
            self.fingers[i].SetValue(True)
            hbox7.Add(self.fingers[i], flag=wx.LEFT|wx.RIGHT|wx.TOP, border=8, proportion=1)
        vbox.Add(hbox7)

        vbox.Add((-1, 5))

        # Enter destination of output file
        hbox8 = wx.BoxSizer(wx.HORIZONTAL)
        st8 = wx.StaticText(panel, label='Enter destination of output file:')
        st8.SetFont(font)
        hbox8.Add(st8, flag=wx.LEFT|wx.RIGHT|wx.TOP, border=8)
        btn_dest = wx.Button(panel, label='Choose file', size=(70, 30))
        hbox8.Add(btn_dest)
        self.Bind(wx.EVT_BUTTON, self.OnSelectDest, btn_dest)
        vbox.Add(hbox8)

        # Indication if file has been added
        hbox_brf = wx.BoxSizer(wx.HORIZONTAL)
        self.st_brf = wx.StaticText(panel, label="Uploaded File: " + self.brfFile)
        self.st_brf.SetFont(font)
        hbox_brf.Add(self.st_brf, flag=wx.LEFT|wx.RIGHT|wx.TOP, border=8)
        vbox.Add(hbox_brf)

        vbox.Add((-1, 5))

        # Enter name of output file
        hbox_outname = wx.BoxSizer(wx.HORIZONTAL)
        self.st_outname = wx.StaticText(panel, label="Enter name of output file: ")
        self.st_outname.SetFont(font)
        hbox_outname.Add(self.st_outname, flag=wx.LEFT|wx.RIGHT|wx.TOP, border=8)
        self.output_name = wx.TextCtrl(panel, value=f"output_{datetime.now()}")
        hbox_outname.Add(self.output_name, flag=wx.LEFT|wx.RIGHT|wx.TOP, border=8)
        vbox.Add(hbox_outname)
        
        vbox.Add((-1, 25))

        ### SECTION 4: Final instructions
        hbox_instruct = wx.BoxSizer(wx.HORIZONTAL)
        self.st_instruct = wx.StaticText(panel, label="After clicking start, please select Braille reading video to track: ")
        self.st_instruct.SetFont(font)
        hbox_instruct.Add(self.st_instruct, flag=wx.LEFT|wx.RIGHT|wx.TOP, border=8)
        vbox.Add(hbox_instruct)

        vbox.Add((-1, 25))

        ### SECTION 5: Start and Close
        hbox_process = wx.BoxSizer(wx.HORIZONTAL)
        btn_start = wx.Button(panel, label='Start', size=(70, 30))
        hbox_process.Add(btn_start)
        self.Bind(wx.EVT_BUTTON, self.StartProcessing, btn_start)

        btn_close = wx.Button(panel, label='Close', size=(70, 30))
        self.Bind(wx.EVT_BUTTON, self.OnQuit, btn_close)

        hbox_process.Add(btn_close, flag=wx.LEFT|wx.BOTTOM, border=5)
        vbox.Add(hbox_process, flag=wx.ALIGN_RIGHT|wx.RIGHT, border=10)

        panel.SetSizer(vbox)

    def OnOpenBrf(self, event):
        """
        Event handler to take a .brf file input
        code from: https://wxpython.org/Phoenix/docs/html/wx.FileDialog.html 
        """
        with wx.FileDialog(self, "Open .brf file", wildcard=".brf files (*.brf)|*.brf", style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return     # the user changed their mind

            # Save file path given
            self.brfFile = fileDialog.GetPath()
        
        # Update text to reflect that file has been uploaded
        # First, extract only the name of file uploaded (and remove the rest of the path)
        brfFileName = re.split('[\\\/]', self.brfFile)
        self.st_brf.SetLabel("File Uploaded: " + brfFileName[-1])

    def OnSelectDest(self, event):
        with wx.DirDialog(self, "Select destination", "", wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST) as dirDialog:
            if dirDialog.ShowModal() == wx.ID_CANCEL:
                return

            # Save directory path given
            self.dirPath = dirDialog.GetPath()
        
        # Update text to reflect that directory has been chosen
        # First, extract only the name of final folder (and remove the rest of the path)
        destNames = re.split('[\\\/]', self.dirPath)
        self.st_brf.SetLabel("Destination Folder: " + destNames[-1])

    def OnQuit(self, event):
        """
        Event handler to quit software
        """
        self.Close()

    def StartProcessing(self, event):
        """
        Event handler to start processing video
        """
        with wx.FileDialog(self, "Open .mp4 file", wildcard=".mp4 files (*.mp4)|*.mp4", style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return     # the user changed their mind

            # Save file path given
            self.videoFile = fileDialog.GetPath()

        # Concatenate given output path and output name into a single file destination
        # TODO might have issues with blackslash on different os? need to double check
        output_path = os.path.join(self.dirPath, self.output_name.GetValue())

        VideoTracker(self.videoFile, page_path=self.brfFile, auto_finger_calib=False, auto_page_calib=self.auto_page.IsChecked(),
                     output_path=output_path, show_frame=self.show_frames.IsChecked()) 
        self.Close()


def main():
    app = wx.App()
    gui = BrailleGUI(None, title='Finger Tracking')
    gui.Show()
    app.MainLoop()


if __name__ == '__main__':
    main()