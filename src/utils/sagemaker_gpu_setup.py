import boto3
import time
from botocore.exceptions import ClientError

class SageMakerGPUManager:
    def __init__(self, region='us-east-1'):
        self.sagemaker = boto3.client('sagemaker', region_name=region)
        self.iam = boto3.client('iam')
        self.region = region
        
    def create_execution_role(self):
        """Create IAM role for SageMaker if it doesn't exist"""
        role_name = 'SageMakerExecutionRole'
        
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "sagemaker.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }]
        }
        
        try:
            role = self.iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=str(trust_policy),
                Description='Execution role for SageMaker'
            )
            
            # Attach necessary policies
            self.iam.attach_role_policy(
                RoleName=role_name,
                PolicyArn='arn:aws:iam::aws:policy/AmazonSageMakerFullAccess'
            )
            
            role_arn = role['Role']['Arn']
            print(f"Created role: {role_arn}")
            time.sleep(10)  # Wait for role to propagate
            
        except ClientError as e:
            if 'EntityAlreadyExists' in str(e):
                role = self.iam.get_role(RoleName=role_name)
                role_arn = role['Role']['Arn']
                print(f"Using existing role: {role_arn}")
            else:
                raise
        
        return role_arn
    
    def create_notebook_instance(self, instance_name='gpu-segmentation-notebook', volumesize=50):
        """Create a SageMaker notebook instance with GPU"""
        
        role_arn = self.create_execution_role()
        
        try:
            response = self.sagemaker.create_notebook_instance(
                NotebookInstanceName=instance_name,
                InstanceType='ml.g4dn.xlarge',  # GPU instance type
                RoleArn=role_arn,
                VolumeSizeInGB=volumesize,
                DirectInternetAccess='Enabled',
                RootAccess='Enabled',
                Tags=[
                    {'Key': 'Purpose', 'Value': 'Fashion-Segmentation'},
                    {'Key': 'Environment', 'Value': 'Development'}
                ]
            )
            
            print(f"Creating notebook instance '{instance_name}'...")
            print("This will take 5-10 minutes...")
            
            # Wait for notebook to be ready
            waiter = self.sagemaker.get_waiter('notebook_instance_in_service')
            waiter.wait(NotebookInstanceName=instance_name)
            
            print(f"Notebook instance '{instance_name}' is ready!")
            
            # Get notebook URL
            url = self.get_notebook_url(instance_name)
            print(f"\nNotebook URL: {url}")
            
            return instance_name, url
            
        except ClientError as e:
            if 'ValidationException' in str(e):
                print(f"Instance '{instance_name}' may already exist.")
                url = self.get_notebook_url(instance_name)
                return instance_name, url
            else:
                raise
    
    def get_notebook_url(self, instance_name):
        """Get the presigned URL for the notebook"""
        response = self.sagemaker.create_presigned_notebook_instance_url(
            NotebookInstanceName=instance_name
        )
        return response['AuthorizedUrl']
    
    def stop_notebook_instance(self, instance_name):
        """Stop the notebook instance (saves costs)"""
        print(f"Stopping notebook instance '{instance_name}'...")
        self.sagemaker.stop_notebook_instance(
            NotebookInstanceName=instance_name
        )
        
        waiter = self.sagemaker.get_waiter('notebook_instance_stopped')
        waiter.wait(NotebookInstanceName=instance_name)
        print("Notebook instance stopped.")
    
    def start_notebook_instance(self, instance_name):
        """Start a stopped notebook instance"""
        print(f"Starting notebook instance '{instance_name}'...")
        self.sagemaker.start_notebook_instance(
            NotebookInstanceName=instance_name
        )
        
        waiter = self.sagemaker.get_waiter('notebook_instance_in_service')
        waiter.wait(NotebookInstanceName=instance_name)
        
        url = self.get_notebook_url(instance_name)
        print(f"Notebook instance is running!")
        print(f"URL: {url}")
        return url
    
    def delete_notebook_instance(self, instance_name):
        """Delete the notebook instance"""
        print(f"Deleting notebook instance '{instance_name}'...")
        
        # Stop first if running
        try:
            self.stop_notebook_instance(instance_name)
        except:
            pass
        
        self.sagemaker.delete_notebook_instance(
            NotebookInstanceName=instance_name
        )
        print("Notebook instance deleted.")
    
    def list_notebook_instances(self):
        """List all notebook instances"""
        response = self.sagemaker.list_notebook_instances()
        
        instances = []
        for nb in response['NotebookInstances']:
            instances.append({
                'Name': nb['NotebookInstanceName'],
                'Status': nb['NotebookInstanceStatus'],
                'InstanceType': nb['InstanceType'],
                'CreationTime': nb['CreationTime']
            })
        
        return instances


# Example usage
if __name__ == "__main__":
    manager = SageMakerGPUManager(region='us-east-1')
    
    # Create notebook instance
    instance_name, url = manager.create_notebook_instance('fashion-segmentation-nb')
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print(f"1. Click this URL to open Jupyter: {url}")
    print("2. Create a new notebook with 'conda_pytorch_p310' kernel")
    print("3. Paste your segmentation script into the notebook")
    print("4. Upload your image to the notebook")
    print("5. Run the cells!")
    print("\nTo check GPU availability, run in notebook:")
    print("   import torch")
    print("   print(torch.cuda.is_available())")
    print("="*70)
    
    # When done, stop or delete
    # manager.stop_notebook_instance(instance_name)  # Stops but keeps it
    # manager.delete_notebook_instance(instance_name)  # Deletes completely