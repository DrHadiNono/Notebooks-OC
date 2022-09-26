import os

meta_dirs = ['Training/', 'Test/']
dir_paths = ['F:/fruits-360/']
s3_path = 's3://dr.hadinono/OC/P8/fruits-360/'

aws_cmd = set()

for meta_dir in meta_dirs:
    for path in dir_paths:
        for name in os.listdir(path+meta_dir):
            if ' ' in name:
                os.rename(path+meta_dir+name, path+meta_dir+name.replace(' ', '_'))
                aws_cmd.add('aws s3 mv "' + s3_path+meta_dir+name +
                            '/" "'+s3_path+meta_dir+name.replace(' ', '_')+'/" --recursive')

print("\n".join(sorted(aws_cmd)))
