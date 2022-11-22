echo 'download PA data..'
curl -O https://datashare.ed.ac.uk/bitstream/handle/10283/3336/PA.zip
echo 'download PA data completed.'
echo
echo 'checking md5sum..'
md5sum -c PA.zip.md5sum
echo
echo 'unzip PA.zip..'
unzip -q PA.zip
echo
echo 'mkdir ../ASVspoof2019, move PA.zip and PA into it.'
mkdir ../ASVspoof2019
mv PA.zip PA ../ASVspoof2019
