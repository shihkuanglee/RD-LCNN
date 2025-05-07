echo 'download PA (2021) data..'
curl -O https://zenodo.org/records/4834716/files/ASVspoof2021_PA_eval_part00.tar.gz
curl -O https://zenodo.org/records/4834716/files/ASVspoof2021_PA_eval_part01.tar.gz
curl -O https://zenodo.org/records/4834716/files/ASVspoof2021_PA_eval_part02.tar.gz
curl -O https://zenodo.org/records/4834716/files/ASVspoof2021_PA_eval_part03.tar.gz
curl -O https://zenodo.org/records/4834716/files/ASVspoof2021_PA_eval_part04.tar.gz
curl -O https://zenodo.org/records/4834716/files/ASVspoof2021_PA_eval_part05.tar.gz
curl -O https://zenodo.org/records/4834716/files/ASVspoof2021_PA_eval_part06.tar.gz
echo 'download PA (2021) data completed.'
echo
echo 'checking md5sum..'
md5sum -c PA_2021.md5sum
echo
echo 'mkdir ../ASVspoof2021'
mkdir ../ASVspoof2021
echo
echo 'extract ASVspoof2021_PA_eval_part0*.tar.gz..'
cat ASVspoof2021_PA_eval_part0*.tar.gz | tar -xzf - -i -C ../ASVspoof2021
echo
echo 'move ASVspoof2021_PA_eval_part0*.tar.gz into ../ASVspoof2021.'
mv ASVspoof2021_PA_eval_part0*.tar.gz ../ASVspoof2021
