from rest_framework import permissions
from users.models import OutletStaff


class IsOwnerOrReadOnly(permissions.BasePermission):
    """
    Custom permission to only allow owners of an object to edit it.
    """

    def has_object_permission(self, request, view, obj):
        # Read permissions are allowed to any request
        if request.method in permissions.SAFE_METHODS:
            return True

        # Write permissions are only allowed to the owner of the profile
        return obj.user == request.user


class IsVendor(permissions.BasePermission):
    """
    Allows access only to vendor users.
    """

    def has_permission(self, request, view):
        # Check if the user is authenticated and has the is_vendor attribute
        return bool(request.user and request.user.is_authenticated and getattr(request.user, 'is_vendor', False))


class IsStaffOfOutlet(permissions.BasePermission):
    def has_object_permission(self, request, view, obj):
        return OutletStaff.objects.filter(
            user=request.user,
            outlet=obj.outlet
        ).exists()


class IsVendorOrStaff(permissions.BasePermission):
    """
    Allows access to users who are either vendors or staff members of an outlet.
    """

    def has_permission(self, request, view):
        # Check if the user is authenticated
        if not request.user.is_authenticated:
            return False

        # Allow access if the user is a vendor
        if hasattr(request.user, 'is_vendor') and request.user.is_vendor:
            return True

        return False

    def has_object_permission(self, request, view, obj):
        # Check if the user is authenticated
        if not request.user.is_authenticated:
            return False

        # Allow access if the user is a vendor
        if hasattr(request.user, 'is_vendor') and request.user.is_vendor:
            return True

        # Check if the user is a staff member of the outlet
        if isinstance(obj, OutletStaff):
            return OutletStaff.objects.filter(
                user=request.user,
                outlet=obj.outlet
            ).exists()

        return False