mkdir HD_UHD_Eyetracking_Videos
wget -r  -P ./  ftp://ftp.ivc.polytech.univ-nantes.fr/IRCCYN_IVC_HD_UHD_Eyetracking_Videos/Gaze_Data/HD/
mv ftp.ivc.polytech.univ-nantes.fr/IRCCYN_IVC_HD_UHD_Eyetracking_Videos/Gaze_Data/ HD_UHD_Eyetracking_Videos/
rm -rf ftp.ivc.polytech.univ-nantes.fr/

wget -r  -P ./  ftp://ftp.ivc.polytech.univ-nantes.fr/IRCCYN_IVC_HD_UHD_Eyetracking_Videos/Videos/HD/
mv ftp.ivc.polytech.univ-nantes.fr/IRCCYN_IVC_HD_UHD_Eyetracking_Videos/Videos/ HD_UHD_Eyetracking_Videos/
rm -rf ftp.ivc.polytech.univ-nantes.fr/

exit
