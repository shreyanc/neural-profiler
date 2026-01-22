import boto3
import modal
import os
import json

app = modal.App("oidc-token-test")

boto3_image = modal.Image.debian_slim().pip_install("boto3")

# Trade a Modal OIDC token for AWS credentials
def get_s3_client(role_arn):
    sts_client = boto3.client("sts")

    # Assume role with Web Identity
    credential_response = sts_client.assume_role_with_web_identity(
        RoleArn=role_arn, RoleSessionName="OIDCSession", WebIdentityToken=os.environ["MODAL_IDENTITY_TOKEN"]
    )

    # Extract credentials
    credentials = credential_response["Credentials"]
    return boto3.client(
        "s3",
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
    )

# List the contents of an S3 bucket
@app.function(image=boto3_image)
def list_bucket_contents(bucket_name, role_arn):
    s3_client = get_s3_client(role_arn)
    response = s3_client.list_objects_v2(Bucket=bucket_name)
    for obj in response["Contents"]:
        print(f"- {obj['Key']} (Size: {obj['Size']} bytes)")

@app.local_entrypoint()
def main():
    # Replace with the role ARN and bucket name from step 2
    list_bucket_contents.remote("s3-signaltrain", "arn:aws:iam::959894397793:role/ModalOIDCRole")